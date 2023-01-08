import time
import math
import serial


ser = serial.Serial("/dev/ttyACM0", 921600)


def move_servos(x, y):
    string = f"{x:6.2f},{y:6.2f}\n"
    data = string.encode()
    ser.write(data)
    ser.flush()


if __name__ == "__main__":
    i = 0
    while True:
        x, y = map(float, input().split())
        move_servos(x, y)
        # move_servos(20 * math.cos(i), 20 * math.sin(i))
        i += 0.02
        time.sleep(0.01)
