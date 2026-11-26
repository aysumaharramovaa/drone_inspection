from pymavlink import mavutil

master = mavutil.mavlink_connection(
    "udpin:127.0.0.1:14540"
)

print("Hesabat")

master.wait_heartbeat(timeout=30)

print("Dron qosuldu!")
print("System ID:", master.target_system)
print("Component ID:", master.target_component)

while True:
    msg = master.recv_match(
        type="GLOBAL_POSITION_INT",
        blocking=True,
        timeout=5
    )

    if msg:
        print("Latitude:", msg.lat / 1e7)
        print("Longitude:", msg.lon / 1e7)
        print("Altitude:", msg.relative_alt / 1000)
        break