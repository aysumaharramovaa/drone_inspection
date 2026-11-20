import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class Scan:
    t: float
    angles: np.ndarray            # rad, sensor çərçivəsi (CCW)
    ranges: np.ndarray            # m (yanlış/qayıtmayan = inf)
    pose: tuple = None            # (x, y, yaw) — CSV-dən, varsa
    gt: tuple = None              # ground-truth (yalnız sim)

ROUTE = [(8.0, 5.0), (8.0, 10.6), (21.0, 10.6), (21.0, 15.0), (24.0, 15.5), (21.0, 15.0),
         (21.0, 10.6), (15.9, 10.6), (15.9, 6.0), (15.9, 10.6), (8.0, 10.6), (8.0, 5.0), (8.0, 1.5)]


class SimLidarSource:
    """drone_inspection_sim.World daxilində uçuş + səs-küylü 360° lidar."""

    def __init__(self, n=360, max_range=12.0, noise=0.02, dropout=0.01, speed=0.6, seed=0):
        from drone_inspection_sim import World, START
        self.world, self.start = World(), START
        self.n, self.max_range, self.noise, self.dropout, self.speed = n, max_range, noise, dropout, speed
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        x, y, yaw = self.start
        wps, t, dt = list(ROUTE), 0.0, 0.1
        ang = np.linspace(0, 2 * math.pi, self.n, endpoint=False)
        while wps:
            d, hit, _ = self.world.cast(x, y, yaw + ang, self.max_range)
            r = d + self.rng.normal(0, self.noise, self.n)
            r[~hit] = np.inf
            r[self.rng.random(self.n) < self.dropout] = np.inf
            yield Scan(t, ang, r, gt=(x, y, yaw))
            tx, ty = wps[0]
            if math.hypot(tx - x, ty - y) < 0.15:
                wps.pop(0)
                continue
            err = wrap(math.atan2(ty - y, tx - x) - yaw)
            yaw = wrap(yaw + float(np.clip(err, -math.radians(90) * dt, math.radians(90) * dt)))
            v = self.speed * math.cos(err) if abs(err) < math.radians(75) else 0.0
            x, y, t = x + v * dt * math.cos(yaw), y + v * dt * math.sin(yaw), t + dt


class ReplayLidarSource:
    """CSV: scan_id,t,angle_deg,range_m[,x,y,yaw]  (range_m <= 0 və ya boş = qayıtmayan)."""

    def __init__(self, path):
        self.path = path

    def __iter__(self):
        with open(self.path, newline="", encoding="utf-8") as f:
            rows, cur = csv.DictReader(f), None
            buf = []

            def flush():
                a = np.radians([float(r["angle_deg"]) for r in buf])
                r = np.array([float(r["range_m"]) if r["range_m"] not in ("", "nan") else np.inf for r in buf])
                r[r <= 0] = np.inf
                pose = None
                if "x" in buf[0] and buf[0]["x"] not in ("", None):
                    pose = (float(buf[0]["x"]), float(buf[0]["y"]), float(buf[0]["yaw"]))
                return Scan(float(buf[0]["t"]), a, r, pose=pose)

            for row in rows:
                if cur is not None and row["scan_id"] != cur and buf:
                    yield flush()
                    buf = []
                cur = row["scan_id"]
                buf.append(row)
            if buf:
                yield flush()


class RPLidarSource:
    """Canlı RPLidar. pip install rplidar-roboticia"""

    def __init__(self, port, baud=115200):
        from rplidar import RPLidar
        self.lidar = RPLidar(port, baudrate=baud)

    def __iter__(self):
        t0 = time.time()
        try:
            for scan in self.lidar.iter_scans(max_buf_meas=5000):
                a = np.radians([m[1] for m in scan])
                r = np.array([m[2] for m in scan], float) / 1000.0     # mm -> m
                r[r <= 0] = np.inf
                yield Scan(time.time() - t0, a, r)
        finally:
            self.lidar.stop()
            self.lidar.stop_motor()
            self.lidar.disconnect()

def icp(src, dst, init, iters=30, thr0=1.0, thr_min=0.25):
    """src (cari skan nöqtələri) -> dst (keyframe). Qaytarır: (x, y, theta), inlier nisbəti, orta xəta."""
    th, t = init[2], np.array(init[:2], float)
    ratio, err = 0.0, 0.0
    for it in range(iters):
        c, s = math.cos(th), math.sin(th)
        p = src @ np.array([[c, -s], [s, c]]).T + t
        d2 = ((p[:, None, :] - dst[None, :, :]) ** 2).sum(-1)
        j = d2.argmin(1)
        dist = np.sqrt(d2[np.arange(len(p)), j])
        m = dist < max(thr_min, thr0 * 0.85 ** it)
        if m.sum() < 15:
            return None, 0.0, 0.0
        P, Q = src[m], dst[j[m]]
        mp, mq = P.mean(0), Q.mean(0)
        U, _, Vt = np.linalg.svd((P - mp).T @ (Q - mq))
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t_new = mq - R @ mp
        th_new = math.atan2(R[1, 0], R[0, 0])
        conv = abs(th_new - th) < 1e-5 and np.linalg.norm(t_new - t) < 1e-5
        th, t = th_new, t_new
        ratio, err = float(m.mean()), float(dist[m].mean())
        if conv:
            break
    return (t[0], t[1], th), ratio, err


def compose(a, rel):
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + rel[0] * c - rel[1] * s, a[1] + rel[0] * s + rel[1] * c, wrap(a[2] + rel[2]))


class ScanMatcher:
    """Skan -> keyframe uyğunlaşdırması (scan-to-scan-dan az drift edir)."""

    def __init__(self, kf_dist=0.4, kf_rot=math.radians(12)):
        self.pose, self.kf_pts, self.kf_pose, self.rel = (0.0, 0.0, 0.0), None, None, (0.0, 0.0, 0.0)
        self.kf_dist, self.kf_rot = kf_dist, kf_rot
        self.ratios, self.poor = [], 0

    def update(self, pts):
        if self.kf_pts is None:
            self.kf_pts, self.kf_pose = pts, self.pose
            return self.pose
        T, ratio, _ = icp(pts, self.kf_pts, self.rel)
        if T is None or ratio < 0.3:
            self.poor += 1                       # zəif uyğunluq — proqnozu saxla
        else:
            self.rel = T
            self.ratios.append(ratio)
        self.pose = compose(self.kf_pose, self.rel)
        if math.hypot(*self.rel[:2]) > self.kf_dist or abs(self.rel[2]) > self.kf_rot:
            self.kf_pts, self.kf_pose, self.rel = pts, self.pose, (0.0, 0.0, 0.0)
        return self.pose

class OccGrid:
    L_FREE, L_OCC, L_MAX = 0.4, 0.85, 5.0

    def __init__(self, size_m=80.0, res=0.1):
        self.res, self.n = res, int(size_m / res)
        self.o = self.n // 2                      # başlanğıc mövqe xəritənin mərkəzindədir
        self.L = np.zeros((self.n, self.n), np.float32)

    def idx(self, x, y):
        return (np.floor(x / self.res) + self.o).astype(int), (np.floor(y / self.res) + self.o).astype(int)

    def integrate(self, pose, ang, rng, max_range):
        x, y, yaw = pose
        a = yaw + ang
        ts = np.arange(self.res / 2, max_range, self.res / 2)
        xs, ys = x + np.cos(a)[:, None] * ts, y + np.sin(a)[:, None] * ts
        ix, iy = self.idx(xs, ys)
        ok = (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n) & (ts[None, :] < rng[:, None] - self.res)
        self.L[iy[ok], ix[ok]] -= self.L_FREE
        ex, ey = self.idx(x + np.cos(a) * rng, y + np.sin(a) * rng)
        ok = (ex >= 0) & (ex < self.n) & (ey >= 0) & (ey < self.n)
        self.L[ey[ok], ex[ok]] += self.L_OCC
        np.clip(self.L, -self.L_MAX, self.L_MAX, out=self.L)

    def stats(self):
        return dict(free_area_m2=round(float((self.L < -0.5).sum() * self.res ** 2), 1),
                    occupied_cells=int((self.L > 0.5).sum()))

    def render(self, traj, pose, pts_w, size=700):
        img = np.full(self.L.shape, 127, np.uint8)
        img[self.L < -0.5], img[self.L > 0.5] = 255, 0
        nz = np.argwhere(np.abs(self.L) > 0.2)
        if len(nz) == 0:
            return np.full((size, size, 3), 127, np.uint8)
        (y0, x0), (y1, x1) = np.maximum(nz.min(0) - 15, 0), np.minimum(nz.max(0) + 15, self.n - 1)
        crop = cv2.cvtColor(img[y0:y1 + 1, x0:x1 + 1][::-1].copy(), cv2.COLOR_GRAY2BGR)
        sc = min(size / crop.shape[1], size / crop.shape[0])
        crop = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)

        def px(x, y):
            return int((x / self.res + self.o - x0) * sc), int((y1 + 1 - (y / self.res + self.o)) * sc)

        if len(traj) > 1:
            cv2.polylines(crop, [np.array([px(*p[:2]) for p in traj[::3]], np.int32)], False, (255, 140, 40), 1)
        for p in pts_w[::2]:
            cv2.circle(crop, px(*p), 1, (0, 0, 255), -1)
        c = px(*pose[:2])
        cv2.circle(crop, c, 5, (0, 120, 255), -1)
        cv2.line(crop, c, px(pose[0] + 0.8 * math.cos(pose[2]), pose[1] + 0.8 * math.sin(pose[2])), (0, 120, 255), 2)
        return crop


def build_source(a):
    if a.source == "sim":
        return SimLidarSource(max_range=a.range)
    if a.source.startswith("rplidar:"):
        a.flip = True
        return RPLidarSource(a.source.split(":", 1)[1])
    return ReplayLidarSource(a.source)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Lidar ilə xəritələmə + scan-matching")
    ap.add_argument("--source", default="sim", help="sim | rplidar:COM3 | skanlar.csv")
    ap.add_argument("--pose", choices=["scanmatch", "file", "none"], default="scanmatch",
                    help="scanmatch=ICP; file=CSV-dəki x,y,yaw; none=dron hərəkətsiz")
    ap.add_argument("--range", type=float, default=12.0, help="lidarın maksimum məsafəsi (m)")
    ap.add_argument("--min-range", type=float, default=0.15)
    ap.add_argument("--flip", action="store_true", help="bucaqlar saat istiqamətindədirsə")
    ap.add_argument("--angle-offset", type=float, default=0.0, help="lidar quraşdırma bucağı (dərəcə)")
    ap.add_argument("--record", default=None, help="xam skanları CSV-yə yaz (replay üçün)")
    ap.add_argument("--out", default="runs/lidar1")
    ap.add_argument("--grid-size", type=float, default=80.0, help="xəritə ölçüsü (m)")
    ap.add_argument("--res", type=float, default=0.1)
    ap.add_argument("--headless", action="store_true")
    a = ap.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    src, grid, matcher = build_source(a), OccGrid(a.grid_size, a.res), ScanMatcher()
    traj, gt_rel, errs, pts_w = [], [], [], np.zeros((0, 2))
    n_scans, path_len, last_pose, gt0, t_last = 0, 0.0, None, None, 0.0
    rec_f = rec_w = None
    if a.record:
        os.makedirs(os.path.dirname(a.record) or ".", exist_ok=True)
        rec_f = open(a.record, "w", newline="", encoding="utf-8")
        rec_w = csv.writer(rec_f)
        rec_w.writerow(["scan_id", "t", "angle_deg", "range_m"])

    try:
        for sc in src:
            if rec_w:                                         # xam (çevrilməmiş) skanı yaz
                rec_w.writerows([n_scans, f"{sc.t:.3f}", f"{math.degrees(x):.3f}", f"{r:.4f}" if np.isfinite(r) else ""]
                                for x, r in zip(sc.angles, sc.ranges))
            ang = (-sc.angles if a.flip else sc.angles) + math.radians(a.angle_offset)
            ok = np.isfinite(sc.ranges) & (sc.ranges > a.min_range) & (sc.ranges < a.range - 0.05)
            ang, rng = ang[ok], sc.ranges[ok]
            if len(rng) < 20:
                continue
            pts = np.c_[rng * np.cos(ang), rng * np.sin(ang)]
            sub = pts[::max(1, len(pts) // 360)]

            if a.pose == "scanmatch":
                pose = matcher.update(sub)
            elif a.pose == "file" and sc.pose is not None:
                pose = sc.pose
            else:
                pose = (0.0, 0.0, 0.0)

            grid.integrate(pose, ang, rng, a.range)
            traj.append((pose[0], pose[1], pose[2], sc.t))
            if last_pose:
                path_len += math.hypot(pose[0] - last_pose[0], pose[1] - last_pose[1])
            last_pose, t_last = pose, sc.t
            if sc.gt is not None:                             # sim: start çərçivəsinə çevir və müqayisə et
                if gt0 is None:
                    gt0 = sc.gt
                dx, dy = sc.gt[0] - gt0[0], sc.gt[1] - gt0[1]
                c, s = math.cos(-gt0[2]), math.sin(-gt0[2])
                g = (dx * c - dy * s, dx * s + dy * c, wrap(sc.gt[2] - gt0[2]))
                gt_rel.append(g)
                errs.append((math.hypot(pose[0] - g[0], pose[1] - g[1]), abs(wrap(pose[2] - g[2]))))
            n_scans += 1

            if not a.headless and n_scans % 3 == 0:
                c, s = math.cos(pose[2]), math.sin(pose[2])
                pts_w = pts @ np.array([[c, s], [-s, c]]) + np.array(pose[:2])
                cv2.imshow("Lidar map (q = exit)", grid.render(traj, pose, pts_w))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if rec_f:
            rec_f.close()
    cv2.imwrite(os.path.join(a.out, "map.png"), grid.render(traj, last_pose or (0, 0, 0), pts_w, 1000))
    np.savez_compressed(os.path.join(a.out, "map.npz"), logodds=grid.L, res=a.res, origin_idx=grid.o)
    with open(os.path.join(a.out, "trajectory.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "x", "y", "yaw"])
        w.writerows((f"{t:.2f}", f"{x:.3f}", f"{y:.3f}", f"{yaw:.4f}") for x, y, yaw, t in traj)
    rep = dict(source=a.source, pose_mode=a.pose, scans=n_scans, duration_s=round(t_last, 1),
               path_length_m=round(path_len, 1), **grid.stats(),
               scan_match=dict(poor_matches=matcher.poor,
                               mean_inlier_ratio=round(float(np.mean(matcher.ratios)), 3) if matcher.ratios else None))
    if errs:
        e = np.array(errs)
        rep["vs_ground_truth"] = dict(
            final_position_error_m=round(float(e[-1, 0]), 2), rmse_position_m=round(float(np.sqrt((e[:, 0] ** 2).mean())), 2),
            max_position_error_m=round(float(e[:, 0].max()), 2), final_yaw_error_deg=round(math.degrees(e[-1, 1]), 1),
            drift_pct_of_path=round(100 * float(e[-1, 0]) / max(path_len, 1e-6), 1))
    with open(os.path.join(a.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"Fayllar: {a.out}/map.png , map.npz , trajectory.csv , report.json")
    return rep


if __name__ == "__main__":
    main()
