import time

import pigpio
from datetime import datetime

from DHT22 import Sensor
import RPi.GPIO as GPIO
from mongo_manager import MongoManager


def flash():
    # Variable for the GPIO pin number
    LED_pin_red = 18

    # Tell the Pi we are using the breakout board pin numbering
    GPIO.setmode(GPIO.BCM)

    # Set up the GPIO pin for output
    GPIO.setup(LED_pin_red, GPIO.OUT)

    GPIO.output(LED_pin_red, GPIO.HIGH)
    print("On")
    time.sleep(0.1)
    GPIO.output(LED_pin_red, GPIO.LOW)
    print("Off")



def read_and_store_temperature_from_dht(port):
# Intervals of about 2 seconds or less will eventually hang the DHT22.
    INTERVAL = 3
    pi = pigpio.pi()
    s = Sensor(pi, port, LED=16, power=8)
    r = 0
    next_reading = time.time()

    r += 1

    s.trigger()

    time.sleep(3)
    print("x")

    print("r: {}  Humidity: {}  Temperature: {} staleness: {:3.2f}  bad_checksum: {}  short_message: {}  missing_message:  {} sensor_resets: {}".format(
        r, s.humidity(), s.temperature(), s.staleness(),
        s.bad_checksum(), s.short_message(), s.missing_message(),
        s.sensor_resets()))

    next_reading += INTERVAL
    m = MongoManager()
    timestamp = datetime.now()
    temperature = s.temperature()
    humidity = s.humidity()

    d = {'timestamp': timestamp,
         'temperature': temperature,
         'humidity': humidity,
         'port': port
         }

    m.add_temp_reading(d)

    s.cancel()
    pi.stop()
    time.sleep(10)

if __name__ == '__main__':

    read_and_store_temperature_from_dht(port = 22)
    read_and_store_temperature_from_dht(port = 6)
    flash()
    time.sleep(10)
