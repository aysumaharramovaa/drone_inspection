
from pymavlink import mavutil
import time
import math

# ==============================
# 1. MAVLINK CONNECTION
# ==============================

master = mavutil.mavlink_connection(
    "udpin:127.0.0.1:14540"
)

master.wait_heartbeat(timeout=30)

print("Dron qosuldu!")

# ==============================
# 2. MODE CONTROL
# ==============================

def set_mode(mode):
    mode_id = master.mode_mapping()[mode]

    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id
    )

    start = time.time()

    while time.time() - start < 10:
        msg = master.recv_match(
            type="HEARTBEAT",
            blocking=True,
            timeout=2
        )

        if msg and mavutil.mode_string_v10(msg) == mode:
            print(f"{mode} mode aktivdir!")
            return

    raise TimeoutError(f"{mode} mode aktivlesmedi!")


# ==============================
# 3. ARM
# ==============================

def arm():
    print("ARM emri gonderilir...")

    master.arducopter_arm()

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
                return

    raise TimeoutError("ARM ugursuz oldu!")


# ==============================
# 4. TAKEOFF
# ==============================

def takeoff(altitude=3):

    print(f"Takeoff: {altitude} metr")

    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0, 0, 0, 0, 0, 0,
        altitude
    )

    start = time.time()

    while time.time() - start < 60:

        msg = master.recv_match(
            type="GLOBAL_POSITION_INT",
            blocking=True,
            timeout=2
        )

        if msg:
            current_alt = msg.relative_alt / 1000

            print(f"Altitude: {current_alt:.2f} m")

            if current_alt >= altitude - 0.2:
                print("Takeoff tamamlandi!")
                return

    raise TimeoutError("Takeoff timeout!")


# ==============================
# 5. LOCAL POSITION
# ==============================

def get_position():

    msg = master.recv_match(
        type="LOCAL_POSITION_NED",
        blocking=True,
        timeout=5
    )

    if msg is None:
        raise TimeoutError("Position alinmadi!")

    return msg.x, msg.y, msg.z


# ==============================
# 6. WAYPOINT NAVIGATION
# ==============================

def send_waypoint(x, y, z):

    master.mav.set_position_target_local_ned_send(
        0,
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111111000,
        x, y, z,
        0, 0, 0,
        0, 0, 0,
        0, 0
    )


def goto_waypoint(x, y, z):

    print(f"\nGOTO: X={x:.2f}, Y={y:.2f}, Z={z:.2f}")

    start = time.time()

    while time.time() - start < 60:

        # Setpoint-i davamli gonder
        send_waypoint(x, y, z)

        msg = master.recv_match(
            type="LOCAL_POSITION_NED",
            blocking=True,
            timeout=1
        )

        if msg is None:
            continue

        distance = math.sqrt(
            (msg.x - x) ** 2 +
            (msg.y - y) ** 2 +
            (msg.z - z) ** 2
        )

        print(f"Target distance: {distance:.2f} m")

        if distance < 0.5:
            print("Waypoint reached!")
            return

    raise TimeoutError("Waypoint reached deyil!")


# ==============================
# 7. INSPECTION HOLD
# ==============================

def inspect(seconds=3):

    print(f"Inspection: {seconds} saniye")

    x, y, z = get_position()

    start = time.time()

    while time.time() - start < seconds:

        # Dronu movqede saxla
        send_waypoint(x, y, z)

        time.sleep(0.5)

    print("Inspection tamamlandi!")


# ==============================
# 8. LAND AND DISARM
# ==============================

def land():

    print("\nLAND rejimi...")

    set_mode("LAND")

    start = time.time()

    while time.time() - start < 90:

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

            if not armed:
                print("Dron endi!")
                print("DISARM tesdiqlendi!")
                return

    raise TimeoutError("Enis tesdiqlenmedi!")


# ==============================
# 9. MAIN MISSION
# ==============================

def run_mission():

    try:

        print("\nMISSION STARTED")

        set_mode("GUIDED")

        arm()

        takeoff(3)

        # Start position
        start_x, start_y, start_z = get_position()

        print("\nStart position:", start_x, start_y, start_z)

        # 3 inspection waypoints
        waypoints = [
            (start_x + 5, start_y, -3),
            (start_x + 5, start_y + 5, -3),
            (start_x, start_y + 5, -3)
        ]

        for index, wp in enumerate(waypoints, 1):

            print(f"\n========== WAYPOINT {index} ==========")

            goto_waypoint(*wp)

            inspect(3)

        print("\nALL WAYPOINTS COMPLETED!")

    finally:

        # Missiya xeta ile bitsə belə enişi yoxla
        try:
            land()
        except Exception as error:
            print("LAND ERROR:", error)
            print("Mission Planner-dezi yoxla!")



run_mission()
