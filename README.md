# Drone Interior Inspection

Computer-vision + lidar pipeline for a drone that enters an indoor facility and checks the interior.
Everything is first validated in a computer simulation, then moved to real sensors step by step.

> Research / portfolio project. The code does not fly a drone. Real flights inside any facility
> require the owner's authorization and proper safety measures.

## Modules

| File | What it does | Status |
|---|---|---|
| `drone_inspection_sim.py` | 2D indoor simulation: entry through the door, lidar mapping, frontier exploration, vision-based findings (simulated camera + detector), return and landing, JSON report | Tested in simulation |
| `drone_inspect_live.py` | Checks what a **real or simulated drone camera** films (webcam, video file, RTSP/UDP, Tello, Gazebo `gz:` topic): YOLO + ByteTrack, unique objects, snapshots, alerts, expected-items checklist | Pipeline tested on video; YOLO path not yet validated here |
| `lidar_inspect.py` | 2D lidar occupancy mapping with ICP scan matching. Sources: simulation, CSV replay, live RPLidar. Can record real scans for replay | Tested in simulation and CSV replay; RPLidar not yet tested on hardware |

![Simulation](docs/sim_demo.png)

## Install

```
pip install -r requirements.txt
```

## Usage

```
# 1) Full mission simulation (window; q = quit)
python drone_inspection_sim.py
python drone_inspection_sim.py --headless --video runs/sim1/run.mp4

# 2) Real drone footage
pip install ultralytics
python drone_inspect_live.py --source drone.mp4 --weights yolov8n.pt --alert person --expect "person,fire extinguisher"
# or describe what to look for in plain text (open-vocabulary YOLO-World); the report lists what is present / missing
python drone_inspect_live.py --source drone.mp4 --vocab "fire extinguisher,person,door,box"

# 2b) Gazebo (gz-sim) drone camera: sudo apt install python3-gz-transport13 python3-gz-msgs10
python drone_inspect_live.py --source gz:/world/warehouse/model/x500_0/link/base_link/sensor/front_camera/image --vocab "box,person"

# 3) Lidar
python lidar_inspect.py --source sim                                   # self-test, reports drift vs ground truth
python lidar_inspect.py --source rplidar:COM3 --record runs/field1/scans.csv   # record on site
python lidar_inspect.py --source runs/field1/scans.csv                 # replay and tune
```

Angle convention for lidar CSV: 0 deg = drone nose, counter-clockwise positive.
RPLidar measures clockwise, so `--flip` is enabled automatically; use `--angle-offset` for mounting rotation.

## Results (simulation)

- Mission: 7/7 objects found, 100% coverage, 0 collisions, 87 s, 66 m path, localization error 2-11 cm.
- Lidar mapping with scan matching (70 m route, noisy simulated lidar): final position error 0.14 m,
  max 0.81 m, mean ICP inlier ratio 0.98.

![Lidar map](docs/lidar_map.png)

## Limitations

- Simulation is 2D with a fixed flight height.
- The simulation detector is color-based and works only on synthetic frames; use `drone_inspect_live.py` with a real model for real footage.
- A model trained on aerial data (e.g. VisDrone) is weak indoors; use a COCO model or fine-tune on indoor data.
- ICP scan matching can drift along long featureless corridors. For real use, add drone odometry (optical flow/IMU)
  or full SLAM (`--pose file` accepts an external pose).

## Roadmap

- Fuse camera findings with lidar pose (place detections on the lidar map)
- ROS 2 / Gazebo or AirSim integration
- Edge deployment of the detector (ONNX FP16/INT8)
