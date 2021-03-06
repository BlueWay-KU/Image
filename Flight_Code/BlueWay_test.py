#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

set_attitude_target.py: (Copter Only)

This example shows how to move/direct Copter and send commands
 in GUIDED_NOGPS mode using DroneKit Python.

Caution: A lot of unexpected behaviors may occur in GUIDED_NOGPS mode.
        Always watch the drone movement, and make sure that you are in dangerless environment.
        Land the drone as soon as possible when it shows any unexpected behavior.

Tested in Python 2.7.10

"""

from dronekit import connect, VehicleMode, LocationGlobal, LocationGlobalRelative
from pymavlink import mavutil # Needed for command message definitions
import time
import math
import sys
sys.path.insert(0, '/home/pi/Bluetooth/iBeacon-Scanner-')
#import testblescan
#import func
#import fireAlarm
import sort
"""
from testVision import dark_channel
from testVision import get_atmo
from testVision import get_trans
from testVision import guided_filter
from testVision import dehaze
from testVision import predict
from testVision import main
"""
import requests
import json
from DTU import FromWeb
import copy
import DTU
import UTE
import locationTest
import DB_load

# Set up option parsing to get connection string
import argparse
parser = argparse.ArgumentParser(description='Control Copter and send commands in GUIDED mode ')
parser.add_argument('--connect',
                   help="Vehicle connection target string. If not specified, SITL automatically started and used.")
args = parser.parse_args()

connection_string = args.connect
sitl = None

# Start SITL if no connection string specified
#if not connection_string:
#    import dronekit_sitl
#    sitl = dronekit_sitl.start_default()
#    connection_string = sitl.connection_string()


# Connect to the Vehicle
print('Connecting to vehicle on: %s' % connection_string)
vehicle = connect(connection_string, wait_ready=True)

def arm_and_takeoff_nogps(aTargetAltitude):
    """
    Arms vehicle and fly to aTargetAltitude without GPS data.
    """

    ##### CONSTANTS #####
    DEFAULT_TAKEOFF_THRUST = 0.7
    SMOOTH_TAKEOFF_THRUST = 0.6

    print("Basic pre-arm checks")
    # Don't let the user try to arm until autopilot is ready
    # If you need to disable the arming check,
    # just comment it with your own responsibility.
    #while not vehicle.is_armable:
    #    print(" Waiting for vehicle to initialise...")
    #    time.sleep(1)


    print("Arming motors")
    # Copter should arm in GUIDED_NOGPS mode
    vehicle.mode = VehicleMode("GUIDED_NOGPS")
    vehicle.armed = True

    while not vehicle.armed:
        print(" Waiting for arming...")
        vehicle.armed = True
        time.sleep(1)

    print("Taking off!")

    thrust = DEFAULT_TAKEOFF_THRUST
    while True:
        current_altitude = vehicle.location.global_relative_frame.alt
        print(" Altitude: %f  Desired: %f" %
              (current_altitude, aTargetAltitude))
        if current_altitude >= aTargetAltitude*0.95: # Trigger just below target alt.
            print("Reached target altitude")
            break
        elif current_altitude >= aTargetAltitude*0.6:
            thrust = SMOOTH_TAKEOFF_THRUST
        set_attitude(thrust = thrust)
        time.sleep(0.2)


def set_attitude(roll_angle = 0.0, pitch_angle = 0.0, yaw_rate = 0.0, thrust = 0.5, duration = 0):
    """
    Note that from AC3.3 the message should be re-sent every second (after about 3 seconds
    with no message the velocity will drop back to zero). In AC3.2.1 and earlier the specified
    velocity persists until it is canceled. The code below should work on either version
    (sending the message multiple times does not cause problems).
    """
    
    """
    The roll and pitch rate cannot be controllbed with rate in radian in AC3.4.4 or earlier,
    so you must use quaternion to control the pitch and roll for those vehicles.
    """
    
    # Thrust >  0.5: Ascend
    # Thrust == 0.5: Hold the altitude
    # Thrust <  0.5: Descend
    msg = vehicle.message_factory.set_attitude_target_encode(
        0, # time_boot_ms
        1, # Target system
        1, # Target component
        0b00000000, # Type mask: bit 1 is LSB
        to_quaternion(roll_angle, pitch_angle), # Quaternion
        0, # Body roll rate in radian
        0, # Body pitch rate in radian
        math.radians(yaw_rate), # Body yaw rate in radian
        thrust  # Thrust
    )
    vehicle.send_mavlink(msg)

    start = time.time()
    while time.time() - start < duration:
        current_altitude = vehicle.location.global_relative_frame.alt
        print(" Altitude: %f  " %current_altitude)
        vehicle.send_mavlink(msg)
        time.sleep(0.1)

def to_quaternion(roll = 0.0, pitch = 0.0, yaw = 0.0):
    """
    Convert degrees to quaternions
    """
    t0 = math.cos(math.radians(yaw * 0.5))
    t1 = math.sin(math.radians(yaw * 0.5))
    t2 = math.cos(math.radians(roll * 0.5))
    t3 = math.sin(math.radians(roll * 0.5))
    t4 = math.cos(math.radians(pitch * 0.5))
    t5 = math.sin(math.radians(pitch * 0.5))

    w = t0 * t2 * t4 + t1 * t3 * t5
    x = t0 * t3 * t4 - t1 * t2 * t5
    y = t0 * t2 * t5 + t1 * t3 * t4
    z = t1 * t2 * t4 - t0 * t3 * t5

    return [w, x, y, z]

while True:
    if int(DB_load.FromWeb(1))!=0:
        break

dtuList = DTU.DTU() #pathFind algorithm result result
print(dtuList)
count = 1
# Take off 2.5m in GUIDED_NOGPS mode.
 
arm_and_takeoff_nogps(0.5)


# Hold the position for 10 seconds.
#print("Hold position for 3 seconds")
# Uncomment the lines below for testing roll angle and yaw rate.
# Make sure that there is enough space for testing this.

# set_attitude(roll_angle = 1, thrust = 0.5, duration = 3)
# set_attitude(yaw_rate = 30, thrust = 0.5, duration = 3)

# Move the drone forward and backward.
# Note that it will be in front of original position due to inertia.
print("Move forward")
a = {0, 0}

while True:
    set_attitude(pitch_angle = -0.7, thrust = 0.5)
    
    droneWitch = locationTest.droneWitch()
    print(droneWitch)
    
    if int(droneWitch) == int(dtuList[count]):
        print("arrived at" + str(droneWitch))
        if count+1 == len(dtuList):
            break
        elif dtuList[count] == 2:
            print("ABC")
            set_attitude(yaw_rate = 30, thrust = 0.5, duration = 3)
            print("DEF")
        elif dtuList[count] == 4 and dtuList[count+1] == 13:
            set_attitude(yaw_rate = 30, thrust = 0.5, duration = 3)
        elif dtuList[count] == 6 and dtuList[count+1] == 7:
            set_attitude(yaw_rate = 30, thrust = 0.5, duration = 3)
        elif dtuList[count] == 10:
            set_attitude(yaw_rate = 30, thrust = 0.5, duration = 3)
        
        count = count + 1

j = 0
while j<300:
    print(j)
    set_attitude(pitch_angle = 1.5, thrust = 0.5)
    print(-j)
    j += 1

set_attitude(yaw_rate = 30, thrust = 0.5, duration = 3)
set_attitude(yaw_rate = 30, thrust = 0.5, duration = 3)

uteList = UTE.UTE()
count = 1


while True:
    set_attitude(pitch_angle = -0.7, thrust = 0.5)
    
    droneWitch = droneWitch()
    
    if droneWitch == uteList[count]:
        print("arrived at" + str(droneWitch))
        if count+1 == len(uteList):
            break
        elif uteList[count] == 10:
            set_attitude(yaw_rate = -30, thrust = 0.5, duration = 3)
        elif uteList[count-1] == 7 and uteList[count] == 6:
            set_attitude(yaw_rate = -30, thrust = 0.5, duration = 3)
#        elif dtuList[count] == 4:
 #           set_attitude(yaw_rate = -30, thrust = 0.5, duration = 3)
  #          num = 0
   #         while True:
    #            if num % 2 == 0:
     #               a = func.system()
      #              print(a)
       #             if a[0] == 1 or num >= 14:
        #                break
         #       set_attitude(pitch_angle = 0.01, thrust = 0.5)
          #      num += 1
       #     set_attitude(yaw_rate = 30, thrust = 0.5, duration = 3)
                #fuckyou
        #        count = 0
        elif dtuList[count] == 2:
            set_attitude(yaw_rate = -30, thrust = 0.5, duration = 3)
        
        count = count + 1



j = 0
while j<400:
    set_attitude(pitch_angle = 1.5, thrust = 0.5)
    j += 1
print("land")
j = 0
print("4")
while j<400:
    set_attitude(thrust = 0.4)
    j += 1
j = 0
print("3")
while j<400:
    set_attitude(thrust = 0.3)
    j += 1
j = 0
print("2")
while j<400:
    set_attitude(thrust = 0.2)
    j += 1
j = 0
print("1")
while j<400:
    set_attitude(thrust = 0.1)
    j += 1
j = 0
print("0")
while j<200:
    set_attitude(thrust = 0.0)
    j += 1

vehicle.mode = VehicleMode("LAND")

#    set_attitude(pitch_angle = -1, thrust = 0.5)
#    if count == 20:
#        count = 0
#    sum = 0
#    isApproach[count] = testblescan.beacon_Result()
#    print(isApproach[count])
#    for i in isApproach:
#        sum += i
#    if sum > 5:
#        break
#    count += 1

#set_attitude(thrust = 0.5)
#print("back")
#while True :
#    set_attitude(pitch_angle = 2, thrust = 0.5)
#    if count == 20:
#        count = 0
#    sum = 0
#    isApproach[count] = testblescan.beacon_Result()
#    print(isApproach[count])
#    for i in isApproach:
#        sum += i
#    if sum <= 2:
#        break
#    count += 1

print("land")
vehicle.mode = VehicleMode("LAND")


#    print(sum)
    # if j==100:
    #     j = 0

    # sum = 0

    # isApproach[j] = beacon_Result()
    # for i in isApproach:
    #     sum += i
    # if sum >= 40:
    #     print("Vehicle arrived at the destination")
    #     vehicle.mode = VehicleMode("LAND")
    #     break
    # j++


print("Setting LAND mode...")
#vehicle.mode = VehicleMode("LAND")
time.sleep(1)

# Close vehicle object before exiting script
print("Close vehicle object")
vehicle.close()

# Shut down simulator if it was started.
if sitl is not None:
    sitl.stop()

print("Completed")
