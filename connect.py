
from pymavlink import mavutil

print("MAVLink yoxlanilir...")

master = mavutil.mavlink_connection(
    "udp:127.0.0.1:14540"
)

for i in range(10):
    msg = master.recv_match(
        type="HEARTBEAT",
        blocking=True,
        timeout=3
    )

    if msg:
        print("\n--- HEARTBEAT ---")
        print("System ID:", msg.get_srcSystem())
        print("Component ID:", msg.get_srcComponent())
        print("Vehicle Type:", msg.type)
        print("Autopilot:", msg.autopilot)
        print("Mode:", mavutil.mode_string_v10(msg))
    else:
        print("Heartbeat yoxdur!")
