import time
import pynput
from pynput.keyboard import Key, Controller
import pyautogui
import matplotlib.pyplot as plt
from HistCorr import HistCorr
from mcpi.minecraft import Minecraft
import pandas as pd
import numpy as np
import cv2



# mcpi
mc = Minecraft.create()
# keyboard
keyboard = Controller()
# mouse
mouse = pynput.mouse.Controller()


def teleport_to_integer_coords():
    # Get the player's tile position
    pos = mc.player.getTilePos()

    # Teleport the player to the coordinate (10, 20, 30)
    mc.player.setTilePos(int(pos.x), int(pos.y), int(pos.z))


def movetime_calculate(distance):
    time.sleep(5)
    teleport_to_integer_coords()
    time.sleep(1)
    pos = mc.player.getTilePos()


    x = np.double(pos.x)
    z = np.double(pos.z)

    x1 = x + distance
    x2 = x - distance
    z1 = z + distance
    z2 = z - distance


    start_time = time.time()  # Set the start time


    while x != x1 and x != x2 and z != z1 and z != z2:
        keyboard.press('w')
        pos = mc.player.getTilePos()
        x = np.double(pos.x)
        z = np.double(pos.z)

        if time.time() - start_time > 20:
            print("Time is up")
            # Release the 'w' key
            keyboard.release('w')
            break

    keyboard.release('w')
    passed_time = time.time() - start_time
    return passed_time, distance/passed_time


# 0 south
def init_direction():
    current_direction = mc.player.getRotation()

    # Define the four directions and their corresponding angles as a list of tuples
    directions = [("South", 0), ("West", 90), ("North", 180), ("East", 270), ("South", 360)]

    # Compute the distance to each direction
    min_distance = float("inf")
    closest_direction = None
    desired_direction = None

    for name, angle_ in directions:
        distance = abs(current_direction - angle_) % 360
        if distance < min_distance:
            min_distance = distance
            closest_direction = name
            desired_direction = angle_

    # print(current_direction)
    # print(desired_direction)


    direction_diff = current_direction - desired_direction
    # print(direction_diff)
    if (direction_diff < 0):
        mouse.move(800/120 * abs(direction_diff), 0)
    else:
        mouse.move(-800/120 * abs(direction_diff), 0)

    
def move1():
    start_time = time.time()
    keyboard.press('w')
    # move 5 meters
    while True:
        if time.time()-start_time > 2.157388210296631/5:
            keyboard.release('w')
            break

def move2():
    start_time = time.time()
    keyboard.press('w')
    # move 10 meters
    while True:
        if time.time()-start_time > 2.157388210296631/2:
            keyboard.release('w')
            break

def turn_left():
    mouse.move(-600, 0)

def turn_right():
    mouse.move(600, 0)

def turn_back():
    mouse.move(1200, 0)




# turn 180 degrees
# mouse move is 800
# time.sleep(3)
# move_smooth(2, 40)
# mouse.move(800, 0)

def move_smooth(xm, t):
    for i in range(t):
        if i < t/2:
            h = i
        else:
            h = t - i
        mouse.move(h*xm, 0)
        time.sleep(1/60)


def resize_image(path):
    image = cv2.imread(path)
    new_size = (480, 360)

    # # Get the original height and width of the image
    # height, width = image.shape[:2]

    # # Define the new dimensions
    # new_height = int(height / 2)
    # new_width = int(width / 2)

    # Resize the image using cv2.resize()
    resized_img = cv2.resize(image, new_size)

    return image, resized_img
   
  
# take screenshot using pyautogui
def take_photo():
    image = pyautogui.screenshot()
    # Convert PIL image to OpenCV format
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)



    # Get the dimensions of the image
    height, width, _ = image.shape

    # Calculate the pixel values to crop
    crop_top = int(height * 0.05)
    crop_bottom = int(height * 0.95)

    # Crop the image
    image = image[crop_top:crop_bottom, :]

    new_size = (480, 360)
    resized_image = cv2.resize(image, new_size)

    return resized_image


def show_picture(image1, image2):
    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Show the original image in the first subplot
    axs[0].imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    axs[0].set_title('image1')

    # Show the resized image in the second subplot
    axs[1].imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    axs[1].set_title('image2')

    # Display the figure
    plt.show()


# test init_direction
def test_init_direction():
    time.sleep(5)
    init_direction()


# test movetime_calculate
def test_movetime_calculate():
    distance = 10
    passed_time, speed = movetime_calculate(distance)
    print("passed time is: " + str(passed_time) + "seconds")
    print("speed is: " + str(speed) + "meters/seconds")


def test():
    # teleport_to_integer_coords()
    # test_init_direction()
    # # move1()
    # # move2()
    time.sleep(1)
    turn_left()
    time.sleep(1)
    turn_right()
    time.sleep(1)
    turn_back()





# the only place call function
# test()



















# from mcpi.minecraft import Minecraft
# from minecraftstuff import MinecraftShape

# serverAddress="192.168.1.94" # change to your minecraft server
# pythonApiPort=4711 #default port for RaspberryJuice plugin is 4711, it could be changed in plugins\RaspberryJuice\config.yml
# playerName="David_Love_Robot" # change to your username

# mc = Minecraft.create()
# pos = mc.player.getPos()

# print("pos: x:{},y:{},z:{}".format(pos.x,pos.y,pos.z))


# myShape = MinecraftShape(mc, pos)


# # create a sign block with the command
# mc.setBlock(pos.x, pos.y, pos.z, 63)
# mc.setSign(pos.x, pos.y, pos.z, 63, 0, "/tp @s 411 70 654 facing 413 70 654")

# execute the command on the sign
# mc.setCommand(pos.x, pos.y, pos.z, "setblock ~ ~ ~ minecraft:air")


# direction = mc.player.getRotation()
# pitch = mc.player.getPitch()


# print("direction:{}".format(direction))
# print("pitch:{}".format(pitch))
# x,y,z = pos = mc.player.getTilePos()
# mc.player.setTilePos(0, 100, 0)
# mc.player.setTilePos(x+2,y,z)

