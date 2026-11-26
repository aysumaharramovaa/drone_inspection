from pymavlink import mavutil

master = mavutil.mavlink_connection(
    "udpin:127.0.0.1:14540"
)

master.wait_heartbeat(timeout=30)

print("Connected!")

while True:
    msg = master.recv_match(
        type="HEARTBEAT",
        blocking=True,
        timeout=5
    )

    if msg:
        print("System:", msg.get_srcSystem())
        print("Component:", msg.get_srcComponent())
        print("Flight mode:", mavutil.mode_string_v10(msg))
        print("Armed:", bool(
            msg.base_mode &
            mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
        ))
        break