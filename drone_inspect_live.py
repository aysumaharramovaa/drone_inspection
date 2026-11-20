#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
drone_inspect_live.py
=====================
REAL dronun kamerasının çəkdiyini yoxlayır (inspection / check-in).
Dron obyektin içinə girəndə kameradan gələn hər şeyi analiz edir:

    kamera/video  ->  YOLO + ByteTrack  ->  unikal obyektlər  ->  snapshot + hesabat

Bu skript dronu İDARƏ ETMİR (uçuşu operator edir). Yalnız görüntünü yoxlayır.
Simulyasiyadakı naviqasiya (drone_inspection_sim.py) ayrıca qalır.

Görüntü mənbələri (--source):
    0, 1, ...                      noutbuk/USB kamerası
    C:\\videolar\\dron_ichəri.mp4     video fayl (dronun yazdığı video)
    rtsp://... / udp://... / http  dron/analoq qəbuledici axını
    tello                          DJI Tello (pip install djitellopy) — yalnız kamera axını,
                                   uçuş komandası göndərilmir

Nümunələr:
    python drone_inspect_live.py --source dron.mp4 --weights yolov8n.pt
    python drone_inspect_live.py --source 0 --weights best.pt --alert person --expect "person,fire extinguisher"
    python drone_inspect_live.py --source udp://@0.0.0.0:11111 --save-video --headless

Pəncərədə 'q' = bitir və hesabat yaz.

Qeyd: VisDrone üzərində öyrədilmiş best.pt hava görüntüsü (yuxarıdan baxış) üçündür,
daxili məkanda zəif işləyə bilər. Daxili yoxlama üçün COCO modeli (yolov8n/s.pt) ilə
başla, sonra öz daxili məkan datanla fine-tune et.
"""
import argparse
import json
import os
import time
import zlib
from dataclasses import dataclass

import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class Track:
    id: int
    label: str
    conf: float
    bbox: tuple          # x1, y1, x2, y2


# ----------------------------------------------------------------------------
# Detektor + tracker
# ----------------------------------------------------------------------------
class YoloTracker:
    """Ultralytics YOLO + ByteTrack. update(frame) -> [Track]"""

    def __init__(self, weights, conf=0.35, imgsz=640, device=None, tracker="bytetrack.yaml"):
        from ultralytics import YOLO
        self.model, self.conf, self.imgsz = YOLO(weights), conf, imgsz
        self.device, self.tracker = device, tracker

    def update(self, frame):
        r = self.model.track(frame, persist=True, tracker=self.tracker, conf=self.conf,
                             imgsz=self.imgsz, device=self.device, verbose=False)[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return []
        ids = b.id.int().tolist() if b.id is not None else [-1] * len(b)
        return [Track(ids[i], self.model.names[k], float(c), tuple(xyxy))
                for i, (xyxy, c, k) in enumerate(zip(b.xyxy.tolist(), b.conf.tolist(),
                                                     b.cls.int().tolist()))]


# ----------------------------------------------------------------------------
# Görüntü mənbəyi
# ----------------------------------------------------------------------------
class Source:
    def __init__(self, src):
        self.tello = None
        if src == "tello":
            from djitellopy import Tello
            self.tello = Tello()
            self.tello.connect()
            print(f"Tello bağlandı, batareya: {self.tello.get_battery()}%")
            self.tello.streamon()
            self.reader = self.tello.get_frame_read()
            self.live, self.fps = True, 30.0
            return
        s = int(src) if src.isdigit() else src
        self.cap = cv2.VideoCapture(s)
        if not self.cap.isOpened():
            raise SystemExit(f"Mənbə açılmadı: {src}")
        self.live = isinstance(s, int) or str(s).lower().startswith(("rtsp", "udp", "http", "tcp"))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        if self.live:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self):
        if self.tello:
            for _ in range(100):
                f = self.reader.frame
                if f is not None:
                    return cv2.cvtColor(f, cv2.COLOR_RGB2BGR)     # djitellopy RGB verir
                time.sleep(0.05)
            return None
        ok, f = self.cap.read()
        return f if ok else None

    def close(self):
        if self.tello:
            self.tello.streamoff()          # yalnız kamera; land/takeoff göndərilmir
        else:
            self.cap.release()


# ----------------------------------------------------------------------------
# Yoxlama məntiqi
# ----------------------------------------------------------------------------
def color_for(label):
    h = zlib.crc32(label.encode()) & 0xFFFFFF
    return (60 + (h & 0xFF) % 180, 60 + ((h >> 8) & 0xFF) % 180, 60 + ((h >> 16) & 0xFF) % 180)


class Inspector:
    def __init__(self, out, alert, expect, confirm_n):
        self.out, self.alert, self.expect, self.confirm_n = out, alert, expect, confirm_n
        self.tracks = {}
        self.alerts = []
        os.makedirs(os.path.join(out, "snapshots"), exist_ok=True)

    def confirmed_ids(self):
        return {i for i, d in self.tracks.items() if d["frames"] >= self.confirm_n}

    def counts(self):
        c = {}
        for d in self.tracks.values():
            if d["frames"] >= self.confirm_n:
                c[d["label"]] = c.get(d["label"], 0) + 1
        return c

    def update(self, vis, tracks, t):
        """vis = bbox-ları çəkilmiş kadr (snapshot üçün)."""
        new = []
        for tr in tracks:
            if tr.id < 0:
                continue
            d = self.tracks.setdefault(tr.id, dict(id=tr.id, label=tr.label, first_seen_s=round(t, 2),
                                                   frames=0, max_conf=0.0, snapshot=None))
            d["frames"] += 1
            d["last_seen_s"] = round(t, 2)
            d["max_conf"] = max(d["max_conf"], round(tr.conf, 3))
            if d["snapshot"] is None and d["frames"] >= self.confirm_n:
                new.append(d)
        for d in new:
            name = f"{len(self.tracks_with_snapshots()) + 1:03d}_{d['label'].replace(' ', '_')}_id{d['id']}_t{t:.1f}s.jpg"
            d["snapshot"] = os.path.join("snapshots", name)
            cv2.imwrite(os.path.join(self.out, d["snapshot"]), vis)
            tag = ""
            if d["label"].lower() in self.alert:
                self.alerts.append(dict(label=d["label"], id=d["id"], time_s=round(t, 2), snapshot=d["snapshot"]))
                tag = "  <<< ALERT"
            print(f"[{t:7.1f}s] yeni obyekt: {d['label']} (id {d['id']}, conf {d['max_conf']:.2f}){tag}")
        return bool(new)

    def tracks_with_snapshots(self):
        return [d for d in self.tracks.values() if d["snapshot"]]

    def report(self, meta):
        objs = sorted((d for d in self.tracks.values() if d["frames"] >= self.confirm_n),
                      key=lambda d: d["first_seen_s"])
        found = {d["label"].lower() for d in objs}
        checklist = dict(expected=sorted(self.expect), found=sorted(e for e in self.expect if e in found),
                         missing=sorted(e for e in self.expect if e not in found))
        return dict(**meta, unique_objects=self.counts(), objects=objs, alerts=self.alerts,
                    checklist=checklist)


def draw(frame, tracks, confirmed):
    for t in tracks:
        x1, y1, x2, y2 = (int(v) for v in t.bbox)
        col = color_for(t.label)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3 if t.id in confirmed else 1)
        cv2.putText(frame, f"{t.label} #{t.id} {t.conf:.2f}", (x1, max(y1 - 6, 14)), FONT, 0.5, col, 1, cv2.LINE_AA)


def hud(frame, t, fps, counts, alerts):
    txt = f"t={t:6.1f}s  fps={fps:4.1f}  " + "  ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(frame, txt[:140], (6, 17), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if alerts:
        a = alerts[-1]
        cv2.putText(frame, f"ALERT: {a['label']} @ {a['time_s']}s", (6, 46), FONT, 0.6, (0, 0, 255), 2, cv2.LINE_AA)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Real dron kamerasını yoxla (YOLO + ByteTrack)")
    ap.add_argument("--source", default="0", help="kamera indeksi, video fayl, rtsp/udp url və ya 'tello'")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="cpu / 0 (GPU)")
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--stride", type=int, default=1, help="hər N-ci kadrı emal et (zəif CPU üçün 2-3)")
    ap.add_argument("--max-width", type=int, default=960)
    ap.add_argument("--confirm", type=int, default=4, help="obyekt neçə kadrda görünəndən sonra təsdiqlənir")
    ap.add_argument("--alert", default="", help="xəbərdarlıq siniflər, vergüllə: person,knife")
    ap.add_argument("--expect", default="", help="yoxlama siyahısı (checklist): 'fire extinguisher,person'")
    ap.add_argument("--out", default="runs/live1")
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--headless", action="store_true")
    a = ap.parse_args(argv)

    split = lambda s: {x.strip().lower() for x in s.split(",") if x.strip()}
    det = YoloTracker(a.weights, a.conf, a.imgsz, a.device, a.tracker)
    src = Source(a.source)
    insp = Inspector(a.out, split(a.alert), split(a.expect), a.confirm)
    writer, tracks, fps, idx, t0 = None, [], 0.0, 0, time.time()
    last_t = t0

    try:
        while True:
            frame = src.read()
            if frame is None:
                break
            if frame.shape[1] > a.max_width:
                s = a.max_width / frame.shape[1]
                frame = cv2.resize(frame, (a.max_width, int(frame.shape[0] * s)))
            t = (time.time() - t0) if src.live else idx / src.fps

            if idx % a.stride == 0:
                tracks = det.update(frame)
                now = time.time()
                fps = 0.9 * fps + 0.1 / max(now - last_t, 1e-6) * a.stride if fps else 1 / max(now - last_t, 1e-6)
                last_t = now
            vis = frame.copy()
            draw(vis, tracks, insp.confirmed_ids())
            if idx % a.stride == 0:
                insp.update(vis, tracks, t)
            hud(vis, t, fps, insp.counts(), insp.alerts)

            if a.save_video:
                if writer is None:
                    os.makedirs(a.out, exist_ok=True)
                    writer = cv2.VideoWriter(os.path.join(a.out, "annotated.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), min(src.fps, 30),
                                             (vis.shape[1], vis.shape[0]))
                writer.write(vis)
            if not a.headless:
                cv2.imshow("Drone interior check (q = exit)", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            idx += 1
    finally:
        src.close()
        if writer:
            writer.release()

    meta = dict(source=a.source, weights=a.weights, duration_s=round(t if idx else 0.0, 1),
                frames_read=idx, confirm_frames=a.confirm)
    rep = insp.report(meta)
    with open(os.path.join(a.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)

    print("\n=== YOXLAMA HESABATI ===")
    print(f"müddət: {rep['duration_s']} s, kadr: {idx}")
    for k, v in sorted(rep["unique_objects"].items()):
        print(f"  {k:<20} {v}")
    if rep["checklist"]["expected"]:
        print(f"checklist: tapıldı={rep['checklist']['found']}  YOXDUR={rep['checklist']['missing']}")
    if rep["alerts"]:
        print(f"ALERT sayı: {len(rep['alerts'])}")
    print(f"Fayllar: {a.out}/report.json , {a.out}/snapshots/")
    return rep


if __name__ == "__main__":
    main()
