
from pymavlink import mavutil
import time

master = mavutil.mavlink_connection(
    "udpin:127.0.0.1:14540"
)

master.wait_heartbeat(timeout=30)
print("Dron qosuldu!")

# GUIDED mode
master.set_mode("GUIDED")
time.sleep(2)

# ARM
print("ARM emri gonderilir...")

master.arducopter_arm()

# ARM statusunu yoxla
armed = False
start = time.time()

while time.time() - start < 15:
    msg = master.recv_match(
        type="HEARTBEAT",
        blocking=True,
        timeout=2
    )

    if msg:
        armed = bool(
            msg.base_mode &
            mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
        )

        if armed:
            print("Dron ARM edildi!")
            break

if not armed:
    raise TimeoutError("Dron ARM edilmedi!")

# TAKEOFF
print("Takeoff baslayir...")

master.mav.command_long_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
    0,
    0, 0, 0, 0, 0, 0, 3
)

# Hündürlüyü yoxla
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
            print("Takeoff tamamlandi!")
            break
