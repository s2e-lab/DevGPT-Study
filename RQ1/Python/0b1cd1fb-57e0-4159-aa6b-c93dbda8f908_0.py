import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threading
import serial
import time

# Global değişkenler
roll = 0
pitch = 0
yaw = 0
ser = None

# Fonksiyonu seri iletişimden veri alacak şekilde güncelle
def update_euler_angles():
    global roll, pitch, yaw, ser
    while True:
        if ser:
            try:
                # Seri porttan veriyi oku (örneğin, "roll,pitch,yaw" formatında)
                data = ser.readline().decode('utf-8').strip().split(',')
                roll = float(data[0])
                pitch = float(data[1])
                yaw = float(data[2])
            except Exception as e:
                print(f"Hata: {e}")

# Veriyi görselleştir
def visualize_euler_angles():
    while True:
        # Veriyi görselleştir
        visualize(roll, pitch, yaw)
        time.sleep(1)  # 1 saniyede bir güncelle

# Veriyi görselleştiren orijinal fonksiyon
def visualize(roll, pitch, yaw):
    # (Daha önce verilen kod ile aynı)

# Seri portu başlat
def start_serial_communication(port, baud_rate):
    global ser
    ser = serial.Serial(port, baud_rate)

# Ana iş parçacığı
if __name__ == "__main__":
    # Seri portu ve baud hızını ayarla (örneğin, "/dev/ttyUSB0" ve 9600)
    serial_port = "/dev/ttyUSB0"
    baud_rate = 9600

    # Seri iletişim iş parçacığını başlat
    serial_thread = threading.Thread(target=update_euler_angles)
    serial_thread.daemon = True
    serial_thread.start()

    # Veri görselleştirme iş parçacığını başlat
    visualization_thread = threading.Thread(target=visualize_euler_angles)
    visualization_thread.daemon = True
    visualization_thread.start()

    # Seri portu başlat
    start_serial_communication(serial_port, baud_rate)

    # Ana iş parçacığını çalışır halde tut
    while True:
        pass
