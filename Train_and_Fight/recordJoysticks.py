'''
This should be run together with the fighting script to record the screenshots and joysticks (if any).
'''
import os
import time
import numpy as np
import pygame
import pickle
import pathlib
import keyboard
import win32com, win32gui, win32con
import win32com.client
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

from time import sleep
from PIL import Image, ImageGrab

def enum_cb(hwnd, results):
    winlist.append((hwnd, win32gui.GetWindowText(hwnd)))
toplist, winlist = [], []
win32gui.EnumWindows(enum_cb, toplist)
handle = [(hwnd, title) for hwnd, title in winlist if 'samurai shodown' in title.lower()][0][0]

global bgn_flag, end_flag, close_flag
bgn_flag, end_flag, close_flag = 0, 0, 0

def plotJoystick(this_time, this_axes, this_buts):
    matplotlib.use("TKAgg")

    times, axes, buts = this_time, np.round(np.array(this_axes[0])[:,:2],2), this_buts[0]
    axes = axes * np.array(np.abs(axes)>=0.5, int)
    axes = axes / (np.linalg.norm(axes, axis=1).reshape(-1,1) + 1e-3)
    idle = np.array(np.sum(np.abs(axes), axis=1) == 0, bool)
    # angle = np.arctan2(axes[:,1], axes[:,0])

    u   = np.array([ 0, 1]).reshape(2,1)
    d   = np.array([ 0,-1]).reshape(2,1)
    r   = np.array([-1, 0]).reshape(2,1)
    l   = np.array([ 1, 0]).reshape(2,1)
    u_l = (u+l)/np.sqrt(2)
    u_r = (u+r)/np.sqrt(2)
    d_l = (d+l)/np.sqrt(2)
    d_r = (d+r)/np.sqrt(2)
    directions = [u, u_r, r, d_r, d, d_l, l, u_l]

    temp = np.hstack([ np.dot(axes, direction)/(np.linalg.norm(axes) + 1e-5).reshape(-1,1) for direction in directions ])
    temp = np.argmin(temp, axis=1)
    temp[idle] = -1

    x_major_locator = MultipleLocator(1)
    y_major_locator = MultipleLocator(1)
    plt.subplot(2,1,1)
    plt.plot(times, temp)
    plt.ylabel("axis input")
    plt.xlim(0, np.floor(max(times)))
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)

    x_major_locator = MultipleLocator(1)
    y_major_locator = MultipleLocator(1)
    plt.subplot(2,1,2)
    plt.plot(times, buts)
    plt.ylabel("button input")
    plt.xlabel("time (s)")
    plt.xlim(0, np.floor(max(times)))
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)

    plt.show()

def keyboardListening(keyboard_event):
    global bgn_flag, end_flag, close_flag
    z = keyboard.KeyboardEvent('down', 28, 'z')
    x = keyboard.KeyboardEvent('down', 28, 'x')
    c = keyboard.KeyboardEvent('down', 28, 'c')
    if (keyboard_event.event_type == 'down') and (keyboard_event.name == z.name):
        bgn_flag = 1
    if (keyboard_event.event_type == 'down') and (keyboard_event.name == x.name):
        end_flag = 1
    if (keyboard_event.event_type == 'down') and (keyboard_event.name == c.name):
        close_flag = 1

print("## LISTENING NOW TO THE KEYBOARD ##")
keyboard.hook(keyboardListening)

print("Tap Z to start keeping the screenshots for the current round.")
print("Tap X to end keeping the screenshots for the current round.")
print("Tap C to end this program.")

pygame.init()
pygame.joystick.init()
joystick_count = pygame.joystick.get_count()
print("Number of joysticks found:", joystick_count)

times, axes, buts = [], [], []
new_start = 0
record_index = len(os.listdir("./_SamuraiJoysticks/")) + 1

while (close_flag != 1):
    if (bgn_flag == 1) and (end_flag == 0):
        if new_start == 0:
            print("START")
            pathlib.Path("./_SamuraiScreenshots/" + str(record_index).rjust(5, "0")).mkdir(parents=True, exist_ok=True) 
            start_time = time.perf_counter()
            pygame.event.get()
            joystick_count = pygame.joystick.get_count()
            temp_times = []
            temp_axes = [[]]*joystick_count
            temp_buts = [[]]*joystick_count
            new_start = 1
            img_index = 0

        pygame.event.get()
        for i in range(joystick_count):
            joystick = pygame.joystick.Joystick(i)
            # joystick.init()
            temp_times.append(time.perf_counter() - start_time)
            temp_axes[i].append([joystick.get_axis(j) for j in range(joystick.get_numaxes())])
            temp_buts[i].append([joystick.get_button(j) for j in range(joystick.get_numbuttons())])

        bbox = win32gui.GetWindowRect(handle)
        temp = ImageGrab.grab(bbox)
        temp.save("./_SamuraiScreenshots/" + str(record_index).rjust(5, "0") + "/" + str(img_index).rjust(5, "0") + "_" + str(time.perf_counter() - start_time) + ".png")
        img_index += 1

        sleep(0.0125)

    if (bgn_flag == 1) and (end_flag == 1):
        print("PAUSE\n")
        bgn_flag = 0
        end_flag = 0
        times.append(temp_times)
        axes.append(temp_axes)
        buts.append(temp_buts)
        new_start = 0
        np.save("./_SamuraiJoysticks/{}.npy".format(str(record_index).rjust(5, "0")), [temp_times, temp_axes, temp_buts], allow_pickle=True)
        # plotJoystick(temp_times, temp_axes, temp_buts)
        record_index += 1

print("ENDED")