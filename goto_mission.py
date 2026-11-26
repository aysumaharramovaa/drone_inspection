
from pymavlink import mavutil
import time

# MAVLink connection
master = mavutil.mavlink_connection(
    "udpin:127.0.0.1:14540"
)

master.wait_heartbeat(timeout=30)
print("Dron qosuldu!")

# GUIDED mode
master.set_mode("GUIDED")
time.sleep(2)

# ARM
print("ARM...")
master.arducopter_arm()
master.motors_armed_wait()

print("Dron ARM edildi!")

# TAKEOFF
print("Takeoff: 3 metr")

master.mav.command_long_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
    0,
    0, 0, 0, 0, 0, 0, 3
)

# 3 metrə çatmağı gözlə
while True:
    msg = master.recv_match(
        type="GLOBAL_POSITION_INT",
        blocking=True,
        timeout=5
    )

    if msg:
        altitude = msg.relative_alt / 1000
        print(f"Altitude: {altitude:.2f} m")

        if altitude >= 2.8:
            break

# Cari lokal mövqeyi al
msg = master.recv_match(
    type="LOCAL_POSITION_NED",
    blocking=True,
    timeout=10
)

start_x = msg.x
start_y = msg.y
start_z = msg.z

print("Start:", start_x, start_y, start_z)

# 5 metr irəli (North)
target_x = start_x + 5
target_y = start_y
target_z = -3

print("GOTO target:", target_x, target_y, target_z)

# GOTO əmri
master.mav.set_position_target_local_ned_send(
    0,
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_FRAME_LOCAL_NED,
    0b0000111111111000,
    target_x,
    target_y,
    target_z,
    0, 0, 0,
    0, 0, 0,
    0, 0
)

# Hədəfə çatmağı gözlə
while True:
    msg = master.recv_match(
        type="LOCAL_POSITION_NED",
        blocking=True,
        timeout=5
    )

    if msg:
        distance = (
            (msg.x - target_x) ** 2 +
            (msg.y - target_y) ** 2 +
            (msg.z - target_z) ** 2
        ) ** 0.5

        print(f"Target distance: {distance:.2f} m")

        if distance < 0.5:
            print("Waypoint reached!")
            break

# LAND
print("LAND...")

master.set_mode("LAND")

print("Mission tamamlandi!")
