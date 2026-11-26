
from pymavlink import mavutil
import time

master = mavutil.mavlink_connection(
    "udpin:127.0.0.1:14540"
)

master.wait_heartbeat(timeout=30)

print("Dron qosuldu!")

# LAND rejimi
mode_id = master.mode_mapping()["LAND"]

master.mav.set_mode_send(
    master.target_system,
    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
    mode_id
)

print("LAND rejimi aktivlesdirilir...")

start = time.time()

while time.time() - start < 90:
    msg = master.recv_match(
        type=["GLOBAL_POSITION_INT", "HEARTBEAT"],
        blocking=True,
        timeout=5
    )

    if not msg:
        continue

    if msg.get_type() == "GLOBAL_POSITION_INT":
        altitude = msg.relative_alt / 1000
        print(f"Altitude: {altitude:.2f} m")

    elif msg.get_type() == "HEARTBEAT":
        armed = bool(
            msg.base_mode &
            mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
        )

        if not armed:
            print("Dron ugurla endi ve DISARM edildi!")
            break
else:
    print("Enis tesdiq olunmadi. Mission Planner-i yoxla.")
