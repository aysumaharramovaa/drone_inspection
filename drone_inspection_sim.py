#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
drone_inspection_sim.py
=======================
Dron ilə qapalı obyektin içinə girib içəridə yoxlama (inspection) aparmaq —
KOMPÜTERDƏ SİMULYASİYA (Gazebo/AirSim-ə keçməzdən əvvəl ilk test).

Pipeline (real dronla eyni məntiq):
    1. Giriş  : APPROACH -> ENTER (qapıdan keçid)
    2. Xəritə : 360° lidar -> occupancy grid (naməlum / boş / divar)
    3. Plan   : frontier exploration + BFS yol axtarışı + yolun sadələşdirilməsi
    4. Görmə  : dron kamerası (sim-də render olunur) -> Detector -> bbox
                bbox-un alt kənarından yer müstəvisi ilə məsafə (pinhole) -> dünya koordinatı
    5. Tracker: eyni obyektin təkrar görünüşlərini birləşdirir (>=3 görünüş = təsdiq)
    6. RETURN : tapılmış xəritə ilə girişə qayıt -> LAND
    7. Hesabat: report.json + map.png (sim-də ground-truth ilə müqayisə)

İstifadə:
    pip install numpy opencv-python
    python drone_inspection_sim.py                      # pəncərə ilə
    python drone_inspection_sim.py --headless --video runs/sim1/run.mp4
    python drone_inspection_sim.py --show-truth         # real divarları solğun göstər

Pəncərədə 'q' = çıxış.

Real sistemə keçid:
    * Camera.render()  ->  real kamera / Gazebo / AirSim frame-i
    * ColorDetector    ->  YoloDetector (öz best.pt modelin)
    * Lidar/occupancy  ->  real lidar / depth kamera / SLAM xəritəsi
    * Mission/FSM və Findings məntiqi dəyişmir.

Qeyd: simulyasiya 2D-dir (hündürlük sabit, ~1.2 m). Məqsəd məntiqi (naviqasiya +
görmə + hesabat) yoxlamaqdır.
"""
import argparse
import json
import math
import os
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# Parametrlər
# ----------------------------------------------------------------------------
RES = 0.1                                   # 1 xana = 0.1 m
W_M, H_M = 32.0, 22.0                       # dünya ölçüsü (m)
GW, GH = int(W_M / RES), int(H_M / RES)
DT = 0.1                                    # simulyasiya addımı (s)

LIDAR_N, LIDAR_R = 240, 8.0                 # şüa sayı, maksimum məsafə (m)
CAM_W, CAM_H = 320, 240
CAM_FOV = math.radians(90)
CAM_R = 10.0                                # kameranın görmə məsafəsi (m)
CAM_Z = 1.2                                 # kamera hündürlüyü (m)
WALL_H = 2.5
F_PX = (CAM_W / 2) / math.tan(CAM_FOV / 2)  # fokus məsafəsi (piksel)

INFLATE = 4                                 # divardan təhlükəsiz məsafə = 4 xana (0.4 m)
V_MAX = 1.0                                 # m/s
YAW_RATE = math.radians(120)
SCAN_RATE = math.radians(120)               # hər dayanacaqda 360° fırlanma = 3 s
VISION_EVERY = 2                            # hər 2 addımda bir kadr emal et (5 Hz)
CONFIRM_N = 3                               # obyekt neçə görünüşdən sonra təsdiqlənir

ENTRY_OUT = (8.0, 1.5)                      # obyektin qarşısı
ENTRY_DOOR = (8.0, 3.1)                     # giriş qapısı
ENTRY_IN = (8.0, 5.0)                       # qapıdan içəri
START = (8.0, 1.0, math.pi / 2)

# sinif: (hündürlük m, en m, BGR rəng)
CLASSES = {
    "person": (1.75, 0.5, (30, 30, 220)),
    "equipment": (1.0, 0.8, (220, 90, 20)),
    "hazard": (0.6, 0.6, (0, 220, 230)),
}


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def to_cell(x, y):
    return int(x / RES), int(y / RES)


def to_world(c):
    return (c[0] + 0.5) * RES, (c[1] + 0.5) * RES


# ----------------------------------------------------------------------------
# Dünya (ground truth)
# ----------------------------------------------------------------------------
class World:
    """Daxili otaqlı bina: 3 cənub otağı, dəhliz, 2 şimal otağı. Giriş cənubdadır."""

    def __init__(self):
        self.grid = np.zeros((GH, GW), np.uint8)
        t = 0.2

        def rect(x0, y0, x1, y1):
            self.grid[int(round(y0 / RES)):int(round(y1 / RES)),
                      int(round(x0 / RES)):int(round(x1 / RES))] = 1

        def hwall(y, x0, x1, gaps=()):
            cur = x0
            for a, b in sorted(gaps):
                rect(cur, y, a, y + t)
                cur = b
            rect(cur, y, x1, y + t)

        def vwall(x, y0, y1, gaps=()):
            cur = y0
            for a, b in sorted(gaps):
                rect(x, cur, x + t, a)
                cur = b
            rect(x, cur, x + t, y1)

        hwall(3, 4, 28, [(7.0, 8.8)])                       # cənub divar + giriş qapısı
        hwall(19 - t, 4, 28)                                # şimal divar
        vwall(4, 3, 19)
        vwall(28 - t, 3, 19)
        hwall(9, 4, 28, [(7.0, 8.8), (15.0, 16.8), (23.0, 24.8)])   # dəhliz-cənub
        hwall(12, 4, 28, [(8.0, 9.8), (20.0, 21.8)])                # dəhliz-şimal
        vwall(12, 3.2, 9)
        vwall(20, 3.2, 9)
        vwall(14, 12.2, 19 - t)

        # yoxlanılacaq obyektlər (sinif, x, y)
        self.objects = [
            ("person", 6.0, 5.5), ("equipment", 16.0, 5.0), ("hazard", 26.0, 7.0),
            ("person", 7.0, 16.0), ("equipment", 24.0, 15.0),
            ("hazard", 18.0, 10.5), ("person", 17.5, 17.0),
        ]

    def collides(self, x, y, r=0.18):
        for dx, dy in ((0, 0), (r, 0), (-r, 0), (0, r), (0, -r)):
            ix, iy = int((x + dx) / RES), int((y + dy) / RES)
            if not (0 <= ix < GW and 0 <= iy < GH) or self.grid[iy, ix]:
                return True
        return False

    def cast(self, x, y, ang, max_r, step=0.05):
        """Vektorlaşdırılmış ray-casting. Qaytarır: (məsafə, toqquşdu?, nümunə-məsafələri)."""
        ts = np.arange(step, max_r + 1e-9, step)
        xs = x + np.cos(ang)[:, None] * ts
        ys = y + np.sin(ang)[:, None] * ts
        ix, iy = (xs / RES).astype(int), (ys / RES).astype(int)
        ok = (ix >= 0) & (ix < GW) & (iy >= 0) & (iy < GH)
        occ = ~ok
        occ[ok] = self.grid[iy[ok], ix[ok]] > 0
        hit = occ.any(1)
        dist = np.where(hit, ts[occ.argmax(1)], max_r)
        return dist, hit, ts


# ----------------------------------------------------------------------------
# Detektorlar (eyni interfeys: detect(frame) -> [Detection])
# ----------------------------------------------------------------------------
@dataclass
class Detection:
    label: str
    conf: float
    bbox: tuple          # x1, y1, x2, y2 (piksel)


class ColorDetector:
    """Simulyasiya üçün: HSV rəng seqmentasiyası + kontur. Real şəkildə işləmir."""
    RANGES = {
        "person": [((0, 150, 100), (8, 255, 255)), ((172, 150, 100), (180, 255, 255))],
        "equipment": [((100, 150, 80), (130, 255, 255))],
        "hazard": [((22, 150, 100), (38, 255, 255))],
    }

    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        out = []
        for label, rngs in self.RANGES.items():
            m = np.zeros(frame.shape[:2], np.uint8)
            for lo, hi in rngs:
                m |= cv2.inRange(hsv, np.array(lo), np.array(hi))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if cv2.contourArea(c) < 40:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                out.append(Detection(label, 0.9, (x, y, x + w, y + h)))
        return out


class YoloDetector:
    """Real kamera / Gazebo / AirSim üçün. pip install ultralytics"""

    def __init__(self, weights, conf=0.4, imgsz=960):
        from ultralytics import YOLO
        self.model, self.conf, self.imgsz = YOLO(weights), conf, imgsz

    def detect(self, frame):
        r = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        return [Detection(self.model.names[int(b.cls)], float(b.conf),
                          tuple(float(v) for v in b.xyxy[0])) for b in r.boxes]


def localize(det, x, y, yaw):
    """bbox-un alt kənarı (yerlə təmas) + kamera hündürlüyü -> dünya koordinatı."""
    x1, y1, x2, y2 = det.bbox
    if y2 >= CAM_H - 2 or y2 <= CAM_H / 2 + 1 or x1 <= 1 or x2 >= CAM_W - 2:
        return None                                # kəsilmiş bbox — etibarsız
    d_perp = F_PX * CAM_Z / (y2 - CAM_H / 2)
    offs = math.atan(((x1 + x2) / 2 - CAM_W / 2) / F_PX)
    d = d_perp / math.cos(offs)
    a = yaw - offs
    return x + d * math.cos(a), y + d * math.sin(a)


class Findings:
    """Eyni obyektin təkrar görünüşlərini birləşdirir (sadə tracker/dedupe)."""

    def __init__(self):
        self.items = []

    def add(self, label, x, y, t, conf):
        for it in self.items:
            if it["label"] == label and math.hypot(it["x"] - x, it["y"] - y) < 1.2:
                n = it["n"]
                it["x"] = (it["x"] * n + x) / (n + 1)
                it["y"] = (it["y"] * n + y) / (n + 1)
                it["conf"] = max(it["conf"], conf)
                it["n"] += 1
                return
        self.items.append(dict(label=label, x=x, y=y, n=1, conf=conf, first_t=round(t, 1)))

    def confirmed(self):
        return [i for i in self.items if i["n"] >= CONFIRM_N]


# ----------------------------------------------------------------------------
# Yol planlama
# ----------------------------------------------------------------------------
NB8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def bfs(trav, start, goal):
    """start-dan ən yaxın goal xanasına 8-qonşu BFS. Qaytarır: xana siyahısı və ya None."""
    q, prev = deque([start]), {start: None}
    while q:
        c = q.popleft()
        if c != start and goal[c[1], c[0]]:
            path = []
            while c is not None:
                path.append(c)
                c = prev[c]
            return path[::-1]
        for dx, dy in NB8:
            n = (c[0] + dx, c[1] + dy)
            if n in prev or not (0 <= n[0] < GW and 0 <= n[1] < GH) or not trav[n[1], n[0]]:
                continue
            if dx and dy and not (trav[c[1], c[0] + dx] and trav[c[1] + dy, c[0]]):
                continue
            prev[n] = c
            q.append(n)
    return None


def los(trav, a, b):
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
    xs = np.linspace(a[0], b[0], n).round().astype(int)
    ys = np.linspace(a[1], b[1], n).round().astype(int)
    return bool(trav[ys, xs].all())


def simplify(path, trav):
    out, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not los(trav, path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out[1:]


# ----------------------------------------------------------------------------
# Simulyasiya + missiya
# ----------------------------------------------------------------------------
class Sim:
    def __init__(self, detector, max_time=600.0, show_truth=False):
        self.world = World()
        self.det = detector
        self.max_time = max_time
        self.show_truth = show_truth
        self.rng = np.random.default_rng(0)

        self.x, self.y, self.yaw = START
        self.known = np.full((GH, GW), -1, np.int8)        # -1 naməlum, 0 boş, 1 divar
        self.lidar_off = np.linspace(0, 2 * math.pi, LIDAR_N, endpoint=False)
        self.geo = np.zeros((GH, GW), bool)                # geofence: yalnız bina daxili
        self.geo[int(3.3 / RES):int(18.7 / RES), int(4.3 / RES):int(27.7 / RES)] = True
        self.free_truth = (self.world.grid == 0) & self.geo
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * INFLATE + 1,) * 2)
        self.blacklist = np.zeros((GH, GW), np.uint8)

        self.findings = Findings()
        self.phase, self.hover = "TAKEOFF", 2.0
        self.wps, self.path_cells, self.target = [], [], None
        self.scan_left, self.blocked = 0.0, False
        self.trav = self.frontier = np.zeros((GH, GW), bool)
        self.coverage, self.dist, self.collisions = 0.0, 0.0, 0
        self.t, self.i = 0.0, 0
        self.trail = []
        self.frame = np.zeros((CAM_H, CAM_W, 3), np.uint8)
        self.dets = []

    # ---- sensorlar -------------------------------------------------------
    def update_map(self):
        ang = self.yaw + self.lidar_off
        dist, hit, ts = self.world.cast(self.x, self.y, ang, LIDAR_R)
        xs = self.x + np.cos(ang)[:, None] * ts
        ys = self.y + np.sin(ang)[:, None] * ts
        ix, iy = (xs / RES).astype(int), (ys / RES).astype(int)
        ok = (ix >= 0) & (ix < GW) & (iy >= 0) & (iy < GH)
        free = ok & ((ts[None, :] < dist[:, None]) | ~hit[:, None])
        self.known[iy[free], ix[free]] = 0
        hx = ((self.x + np.cos(ang) * dist) / RES).astype(int)
        hy = ((self.y + np.sin(ang) * dist) / RES).astype(int)
        m = hit & (hx >= 0) & (hx < GW) & (hy >= 0) & (hy < GH)
        self.known[hy[m], hx[m]] = 1

    def render_camera(self):
        """Birinci şəxs görünüş: divarlar (ray-cast) + obyektlər (billboard) + səs-küy."""
        u = np.arange(CAM_W)
        offs = np.arctan((u + 0.5 - CAM_W / 2) / F_PX)
        dist, hit, _ = self.world.cast(self.x, self.y, self.yaw - offs, CAM_R)
        perp = np.maximum(dist * np.cos(offs), 0.05)
        img = np.empty((CAM_H, CAM_W, 3), np.uint8)
        img[:CAM_H // 2], img[CAM_H // 2:] = (45, 45, 45), (95, 95, 95)
        top = np.clip(CAM_H / 2 - F_PX * (WALL_H - CAM_Z) / perp, 0, CAM_H).astype(int)
        bot = np.clip(CAM_H / 2 + F_PX * CAM_Z / perp, 0, CAM_H).astype(int)
        shade = (230 * (1 - 0.75 * np.minimum(perp / CAM_R, 1))).astype(np.uint8)
        for c in range(CAM_W):
            if hit[c]:
                img[top[c]:bot[c], c] = shade[c]

        vis = []
        for label, ox, oy in self.world.objects:
            dx, dy = ox - self.x, oy - self.y
            d = math.hypot(dx, dy)
            o = wrap(self.yaw - math.atan2(dy, dx))
            if 0.3 < d < CAM_R and abs(o) < CAM_FOV / 2:
                vis.append((d, o, label))
        for d, o, label in sorted(vis, reverse=True):
            h, w, col = CLASSES[label]
            dp = d * math.cos(o)
            cx = CAM_W / 2 + F_PX * math.tan(o)
            y0 = int(CAM_H / 2 - F_PX * (h - CAM_Z) / dp)
            y1 = int(CAM_H / 2 + F_PX * CAM_Z / dp)
            x0, x1 = int(cx - F_PX * w / (2 * dp)), int(cx + F_PX * w / (2 * dp))
            for c in range(max(x0, 0), min(x1, CAM_W)):
                if d < dist[c]:                           # divar arxasında qalan görünmür
                    img[max(y0, 0):min(y1, CAM_H), c] = col
        img = np.clip(img + self.rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
        return img

    def vision_step(self):
        self.frame = self.render_camera()
        self.dets = self.det.detect(self.frame)
        for d in self.dets:
            p = localize(d, self.x, self.y, self.yaw)
            if p is not None:
                self.findings.add(d.label, p[0], p[1], self.t, d.conf)

    # ---- hərəkət ---------------------------------------------------------
    def drive(self, tx, ty, speed):
        des = math.atan2(ty - self.y, tx - self.x)
        err = wrap(des - self.yaw)
        self.yaw = wrap(self.yaw + float(np.clip(err, -YAW_RATE * DT, YAW_RATE * DT)))
        v = speed * math.cos(err) if abs(err) < math.radians(75) else 0.0
        nx, ny = self.x + v * DT * math.cos(self.yaw), self.y + v * DT * math.sin(self.yaw)
        if self.world.collides(nx, ny):
            self.collisions += 1
            self.blocked = True
            return
        self.dist += math.hypot(nx - self.x, ny - self.y)
        self.x, self.y = nx, ny

    def follow(self, speed):
        """Waypoint-lərlə get. Hamısı bitəndə True qaytarır."""
        if not self.wps:
            return True
        tx, ty = self.wps[0]
        if math.hypot(tx - self.x, ty - self.y) < 0.25:
            self.wps.pop(0)
            return not self.wps
        self.drive(tx, ty, speed)
        return False

    # ---- planlama --------------------------------------------------------
    def refresh(self):
        occ = (self.known == 1).astype(np.uint8)
        infl = cv2.dilate(occ, self.kernel)
        self.trav = (self.known == 0) & (infl == 0) & self.geo
        unk = ((self.known == -1) & self.geo).astype(np.float32)
        self.frontier = self.trav & (cv2.blur(unk, (5, 5)) > 0.2)
        self.coverage = float(((self.known == 0) & self.free_truth).sum() / self.free_truth.sum())

    def plan_explore(self):
        self.refresh()
        goal = self.frontier & ~self.blacklist.astype(bool)
        path = bfs(self.trav, to_cell(self.x, self.y), goal)
        if path is None:
            return False
        self.target, self.path_cells = path[-1], path
        self.wps = [to_world(c) for c in simplify(path, self.trav)]
        return True

    def blacklist_target(self):
        if self.target is not None:
            cv2.circle(self.blacklist, self.target, 12, 1, -1)

    def start_return(self):
        self.refresh()
        goal = np.zeros((GH, GW), np.uint8)
        cv2.circle(goal, to_cell(*ENTRY_IN), 3, 1, -1)
        start = to_cell(self.x, self.y)
        path = None if goal[start[1], start[0]] else bfs(self.trav, start, goal.astype(bool))
        if path is None and not goal[start[1], start[0]]:
            path = bfs((self.known == 0) & self.geo, start, goal.astype(bool))   # yumşaldılmış
        self.path_cells = path or []
        self.wps = [to_world(c) for c in simplify(path, self.trav)] if path else [ENTRY_IN]
        self.wps += [ENTRY_DOOR, ENTRY_OUT, START[:2]]
        self.set_phase("RETURN")

    def set_phase(self, p):
        print(f"[{self.t:6.1f}s] {self.phase} -> {p}   (coverage {self.coverage * 100:.0f}%, "
              f"confirmed {len(self.findings.confirmed())})")
        self.phase = p

    # ---- missiya FSM -----------------------------------------------------
    def mission_step(self):
        ph = self.phase
        if ph == "TAKEOFF":
            self.hover -= DT
            if self.hover <= 0:
                self.wps = [ENTRY_OUT]
                self.set_phase("APPROACH")
        elif ph == "APPROACH":
            if self.follow(V_MAX):
                self.wps = [ENTRY_DOOR, ENTRY_IN]
                self.set_phase("ENTER")
        elif ph == "ENTER":
            if self.follow(0.6):
                self.scan_left = 2 * math.pi
                self.set_phase("EXPLORE")
        elif ph == "EXPLORE":
            self.explore_step()
        elif ph == "RETURN":
            if self.follow(V_MAX * 0.8):
                self.set_phase("LAND")
        elif ph == "LAND":
            self.set_phase("DONE")

    def explore_step(self):
        if self.t > self.max_time * 0.85:                   # batareya/vaxt ehtiyatı
            return self.start_return()
        if self.scan_left > 0:                              # dayanacaqda 360° skan
            self.yaw = wrap(self.yaw + SCAN_RATE * DT)
            self.scan_left -= SCAN_RATE * DT
            if self.scan_left <= 0:
                self.refresh()
                if self.target and self.frontier[self.target[1], self.target[0]]:
                    self.blacklist_target()                 # skandan sonra da frontier qalıbsa — keç
                self.wps, self.target = [], None
            return
        if not self.wps:
            if not self.plan_explore():
                return self.start_return()                  # əlçatan frontier qalmadı
        if self.i % 10 == 0 and self.wps and self.target:   # xəritə dəyişibsə yenidən planla
            self.refresh()
            c = to_cell(self.x, self.y)
            if not self.frontier[self.target[1], self.target[0]] or \
                    not los(self.trav, c, to_cell(*self.wps[0])):
                if not self.plan_explore():
                    return self.start_return()
        if self.blocked:
            self.blocked = False
            self.blacklist_target()
            self.wps = []
            return
        if self.follow(V_MAX):
            self.scan_left = 2 * math.pi

    def tick(self):
        self.update_map()
        if self.i % VISION_EVERY == 0:
            self.vision_step()
        self.mission_step()
        self.trail.append((self.x, self.y))
        self.t += DT
        self.i += 1

    # ---- vizuallaşdırma --------------------------------------------------
    def compose(self):
        S = 2
        k = self.known
        img = np.full((GH, GW, 3), 110, np.uint8)
        if self.show_truth:
            img[(self.world.grid == 1) & (k == -1)] = (150, 110, 110)
        img[k == 0], img[k == 1] = (245, 245, 245), (0, 0, 0)
        m = cv2.resize(img[::-1], (GW * S, GH * S), interpolation=cv2.INTER_NEAREST)

        def px(x, y):
            return int(x / RES * S), int((GH - y / RES) * S)

        if len(self.trail) > 1:
            cv2.polylines(m, [np.array([px(*p) for p in self.trail[::5]], np.int32)], False,
                          (255, 160, 60), 1)
        if self.wps:
            pts = [px(self.x, self.y)] + [px(*w) for w in self.wps]
            cv2.polylines(m, [np.array(pts, np.int32)], False, (0, 160, 0), 1)
        for it in self.findings.items:
            col = CLASSES[it["label"]][2]
            conf = it["n"] >= CONFIRM_N
            cv2.circle(m, px(it["x"], it["y"]), 7, col, -1 if conf else 1)
            if conf:
                cv2.putText(m, it["label"], (px(it["x"], it["y"])[0] + 9, px(it["x"], it["y"])[1] + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        c = px(self.x, self.y)
        for s in (-1, 1):
            a = self.yaw + s * CAM_FOV / 2
            cv2.line(m, c, px(self.x + 3 * math.cos(a), self.y + 3 * math.sin(a)), (0, 120, 255), 1)
        cv2.circle(m, c, 5, (0, 120, 255), -1)

        cam = cv2.resize(self.frame, (CAM_W * 2, CAM_H * 2), interpolation=cv2.INTER_NEAREST)
        for d in self.dets:
            x1, y1, x2, y2 = (int(v * 2) for v in d.bbox)
            col = CLASSES.get(d.label, (0, 255, 0))[2]
            cv2.rectangle(cam, (x1, y1), (x2, y2), col, 2)
            cv2.putText(cam, d.label, (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)
        cv2.putText(cam, "DRONE CAMERA", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        canvas = np.full((CAM_H * 2, GW * S + CAM_W * 2, 3), 30, np.uint8)
        canvas[:GH * S, :GW * S] = m
        canvas[:, GW * S:] = cam
        txt = (f"t={self.t:5.1f}s  {self.phase}  coverage={self.coverage * 100:4.1f}%  "
               f"found={len(self.findings.confirmed())}  dist={self.dist:5.1f}m  coll={self.collisions}")
        cv2.putText(canvas, txt, (8, GH * S + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return canvas

    # ---- hesabat ---------------------------------------------------------
    def report(self):
        conf = self.findings.confirmed()
        gt = []
        for label, x, y in self.world.objects:
            ds = [math.hypot(f["x"] - x, f["y"] - y) for f in conf if f["label"] == label]
            best = min(ds) if ds else None
            gt.append(dict(label=label, true_xy=[x, y], found=bool(best is not None and best < 1.5),
                           error_m=None if best is None else round(best, 2)))
        fp = [f for f in conf if not any(
            f["label"] == l and math.hypot(f["x"] - x, f["y"] - y) < 1.5 for l, x, y in self.world.objects)]
        found = sum(g["found"] for g in gt)
        return dict(
            mission_time_s=round(self.t, 1), distance_m=round(self.dist, 1),
            coverage_pct=round(self.coverage * 100, 1), collisions=self.collisions,
            final_phase=self.phase, objects_total=len(gt), objects_found=found,
            false_positives=len(fp),
            findings=[dict(label=f["label"], x=round(f["x"], 2), y=round(f["y"], 2),
                           sightings=f["n"], first_seen_s=f["first_t"]) for f in conf],
            ground_truth_check=gt,
        )


def main():
    ap = argparse.ArgumentParser(description="Dron daxili yoxlama simulyasiyası")
    ap.add_argument("--headless", action="store_true", help="pəncərə açma")
    ap.add_argument("--video", default=None, help="məs. runs/sim1/run.mp4")
    ap.add_argument("--out", default="runs/sim1", help="hesabat qovluğu")
    ap.add_argument("--max-time", type=float, default=600.0, help="missiya vaxt limiti (s)")
    ap.add_argument("--detector", choices=["color", "yolo"], default="color")
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--show-truth", action="store_true", help="real divarları solğun göstər")
    a = ap.parse_args()

    det = ColorDetector() if a.detector == "color" else YoloDetector(a.weights)
    sim = Sim(det, a.max_time, a.show_truth)
    os.makedirs(a.out, exist_ok=True)

    writer = None
    if a.video:
        os.makedirs(os.path.dirname(a.video) or ".", exist_ok=True)
        h, w = sim.compose().shape[:2]
        writer = cv2.VideoWriter(a.video, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))

    while sim.phase != "DONE" and sim.t < a.max_time * 1.5:
        sim.tick()
        if sim.i % 2 == 0 and (writer or not a.headless):
            frame = sim.compose()
            if writer:
                writer.write(frame)
            if not a.headless:
                cv2.imshow("Drone interior inspection (q = exit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    if writer:
        writer.release()

    rep = sim.report()
    with open(os.path.join(a.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    cv2.imwrite(os.path.join(a.out, "map.png"), sim.compose())
    print("\n=== HESABAT ===")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("findings", "ground_truth_check")}, indent=2))
    for g in rep["ground_truth_check"]:
        print(f"  {g['label']:<10} true={g['true_xy']}  found={g['found']}  error={g['error_m']} m")
    print(f"Fayllar: {a.out}/report.json , {a.out}/map.png")
    if not a.headless:
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
