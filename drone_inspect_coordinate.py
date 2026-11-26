import time
from pymavlink import mavutil

master = mavutil.mavlink_connection("udp:127.0.0.1:14540")
master.wait_heartbeat()
print("Dron qoşuldu")

def set_mode(mode):
    mode_id = master.mode_mapping()[mode]
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id)

def arm():
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
    master.motors_armed_wait()

def takeoff(alt=3.0):
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, alt)

def goto_local(x, y, z):
    """Yerli NED koordinat (metr). z mənfi = yuxarı."""
    master.mav.set_position_target_local_ned_send(
        0, master.target_system, master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111111000,   # yalnız mövqe istifadə olunur
        x, y, z, 0, 0, 0, 0, 0, 0, 0, 0)

def get_position():
    msg = master.recv_match(type="LOCAL_POSITION_NED", blocking=True)
    return msg.x, msg.y, msg.z

def reached(target, tol=0.5):
    x, y, z = get_position()
    return abs(x-target[0]) < tol and abs(y-target[1]) < tol and abs(z-target[2]) < tol

def run_mission(target_xyz, detector):
    findings = set()

    set_mode("GUIDED") if "GUIDED" in master.mode_mapping() else set_mode("OFFBOARD")
    arm()
    takeoff(3.0)
    time.sleep(8)

    print("GOTO...")
    goto_local(*target_xyz)
    while not reached(target_xyz):
        time.sleep(0.2)

    print("INSPECT...")
    waypoints = [
        (target_xyz[0] + 2, target_xyz[1],     target_xyz[2]),
        (target_xyz[0] + 2, target_xyz[1] + 2, target_xyz[2]),
        (target_xyz[0],     target_xyz[1] + 2, target_xyz[2]),
    ]
    for wp in waypoints:
        goto_local(*wp)
        t0 = time.time()
        while not reached(wp) and time.time() - t0 < 20:
            findings |= detector.detect_current_frame()   # kameradan detekt
            time.sleep(0.2)

    print("Tapılanlar:", findings)
    return findings