
from pymavlink import mavutil
import time

master = mavutil.mavlink_connection(
    "udpin:127.0.0.1:14540"
)

master.wait_heartbeat(timeout=30)

print("Dron qosuldu!")

# GUIDED rejimini aktiv et
mode_mapping = master.mode_mapping()

print("Movcud rejimler:", mode_mapping)

mode_id = mode_mapping["GUIDED"]

master.mav.set_mode_send(
    master.target_system,
    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
    mode_id
)

# Rejimin dəyişməsini yoxla
for _ in range(10):
    msg = master.recv_match(
        type="HEARTBEAT",
        blocking=True,
        timeout=2
    )

    if msg:
        mode = mavutil.mode_string_v10(msg)
        print("Flight Mode:", mode)

        if mode == "GUIDED":
            print("GUIDED mode aktivdir!")
            break

    time.sleep(0.5)
