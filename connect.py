from pymavlink import mavutil

master = mavutil.mavlink_connection("udp:127.0.0.1:14540")

print("Heartbeat gözlənilir...")

master.wait_heartbeat(timeout=30)

print("Dron qoşuldu!")
print("System ID:", master.target_system)
print("Component ID:", master.target_component)