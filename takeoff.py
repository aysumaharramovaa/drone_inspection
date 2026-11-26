
from pymavlink import mavutil
import time

master = mavutil.mavlink_connection(
    "udpin:127.0.0.1:14540"
)

master.wait_heartbeat(timeout=30)
print("Dron qosuldu!")

# GUIDED rejimi
master.set_mode("GUIDED")
time.sleep(2)

# ARM
print("ARM emri gonderilir...")

master.arducopter_arm()
master.motors_armed_wait(timeout=15)

print("Dron ARM edildi!")

# TAKEOFF
print("Takeoff baslayir...")

master.mav.command_long_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
    0,
    0, 0, 0, 0, 0, 0, 3
)

# 3 metr yukseklik gozle
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
            print("3 metr yukseklik erreicht!")
            break
