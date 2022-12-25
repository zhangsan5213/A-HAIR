'''
Trainer for the AI Opponent, including some basic functions including recognizing the characters.
'''
import time
import pathos
import numpy as np
import win32com, win32gui, win32con
import win32com.client
import pydirectinput
pydirectinput.PAUSE = 0.005

from pydirectinput import press, keyUp, keyDown
from PIL import Image, ImageGrab

from yolo.detect_detail import *
print("\nYOLO loaded for character recognition.\n")

from mpo import MPO
from mpo_nets import CategoricalActor, Critic

def enum_cb(hwnd, results):
    winlist.append((hwnd, win32gui.GetWindowText(hwnd)))
toplist, winlist = [], []
win32gui.EnumWindows(enum_cb, toplist)
handle = [(hwnd, title) for hwnd, title in winlist if 'Samurai Shodown NEOGEO Collection' in title][0][0]

movement_list = ["w", "back", "s", "face"] ## 0~3
fight_list = ["i", "k", "l"] ## 4~9
skill_list = ["s_back+s_back_o", ## snowfall
              "back+s_s_face+s_face_w_o","back+s_s_face+s_face_w_o", ## zeroth swallow
              "w_back_back+s_s_face+s_face_o","w_back_back+s_s_face+s_face_o", ## swallow
              "s_s+face_face_j",
              "s_s+face_face_k_back_back_back_back_back_back_back_back_i_i_i_i_i_back_back_back_back_back_back_back_back+s_s_face+s_face_w_o",
              "s_s+face_face_j_back_back_back_back_back_back_back_back_i_i_i_i_i_back_back_back_back_back_back_back_s_s+face_face_j",
              "s_s+face_face_k_back_back_back_back_back_back_back_back_i_i_i_i_i_back_back_back_back_back_back_back_s_s+face_face_k"] ## ghostly strike

combo_list = ["s_u", "s_i", "s_o", "s_o", "s_k", "s_l", "back_w", "face_w"]
action_list = fight_list + skill_list

class trainer(object):
    def __init__(self, handle, winMark_path, action_list):

        self.handle = handle
        self.winMark = cv2.imread(winMark_path, cv2.COLOR_BGR2RGB)
        self.action_list = action_list
        self.dim_actions = len(action_list)

        self.states = ['1_1', '1_2', '1_3', '1_4', '1_5', '1_back', '1_dashing', '1_downSlash', '1_idle', '1_jump', '1_knocked', '1_parry', '1_upSlash', '2_parry', '2_u', '2_defeated', '2_back', '2_i', '2_dashing', '2_jump', '2_skill3', '2_j', '2_skill2', '2_l', '2_skill1', '2_o', '2_idle', '2_k']
        ## states used to trained the weights
        self.states_detailed = ['1_u', '1_i', '1_o', '1_j', '1_k', '1_upSlash', '1_downSlash', '1_aeroStrike', '1_aeroKick',
                                '1_idle', '1_hit', '1_down', '1_back', '1_parry', '1_jump', '1_dashForward', '1_rollBack', '1_ultimate',
                                '1_crescentMoon', '1_winePot', '1_whirlWind', '1_earthquake',
                                '2_u', '2_i', '2_o', '2_j', '2_k', '2_upSlash', '2_downSlash', '2_aeroStrike', '2_aeroKick',
                                '2_idle', '2_hit', '2_down', '2_back', '2_parry', '2_jump', '2_dashForward', '2_rollBack', '2_ultimate',
                                '2_snowfall', '2_fakeSnowfall', '2_swallowSlash', '2_ghostlyStrike',
                                '1_meleeThrow', '1_rangeThrow', '1_parried',
                                '2_meleeThrow', '2_rangeThrow', '2_parried',
                                '1_faint', '2_faint']
        ## states used for detailed analysis
        self.sizes = np.load("./data_and_models/templateSizes.npy", allow_pickle=True).tolist()
        self.sizes_detailed = np.load("./data_and_models/detailedSizes.npy", allow_pickle=True).tolist()
        
        self.hp1 = 100
        self.hp2 = 100

        self.position_1, self.position_2 = [-1, -1], [-1, -1]
        self.label_1, self.label_2 = -1, -1
        self.box_size_1, self.box_size_2 = -1, -1
        ## if cannot recognize, default as -1

    def inputAction(self, action_index, face, back):
        combination = self.action_list[action_index].replace("face", face).replace("back", back)
        keys = combination.split("_")
        keyUp(back)
        for key in keys:
            if "[" in key:
                temp_split = key.split("[")
                press(temp_split[0])
                time.sleep(float(temp_split[1][:-1]))
            elif "+" in key:
                temp_key_1, temp_key_2 = key.split("+")
                keyDown(temp_key_1)
                keyDown(temp_key_2)
                keyUp(temp_key_1)
                keyUp(temp_key_2)
            else:
                press(key)

        keyDown(back)

    def findState(self):
        ## use the YOLO model to get the players' information
        temp_image = self.game_img

        pred_labels, pred_conf, pred_boxes = readScreenShot4States_detailed(temp_image)

        flag_1, flag_2 = 0, 0
        ## flag for finding the boxes, default as 0

        for i in range(len(pred_labels)):
            if (flag_1 == 1) and (flag_2 == 1): break

            if (flag_1 == 0) and (self.states[pred_labels[i]][0] == "2"):
                self.position_1 = [(pred_boxes[i][1]+pred_boxes[i][3])/2,
                                   (pred_boxes[i][0]+pred_boxes[i][2])/2]
                              ## horizontal and vertical positions
                self.label_1 = pred_labels[i]
                self.box_size_1 = (pred_boxes[i][3]-pred_boxes[i][1]) * (pred_boxes[i][2]-pred_boxes[i][0])
                flag_1 = 1

            if (flag_2 == 0) and (self.states[pred_labels[i]][0] == "1"):
                self.position_2 = [(pred_boxes[i][1]+pred_boxes[i][3])/2,
                                   (pred_boxes[i][0]+pred_boxes[i][2])/2]
                self.label_2 = pred_labels[i]
                self.box_size_2 = (pred_boxes[i][3]-pred_boxes[i][1]) * (pred_boxes[i][2]-pred_boxes[i][0])
                flag_2 = 1

    def findHPs(self):
        ## health points of the two players as reward
        line = np.asarray(self.game_img)[:, :, :][99, :]
        hp1_color, hp2_color = line[290], line[365]

        temp_1 = np.flip(line[ 40:295], axis=0)
        norm_1 = np.linalg.norm(temp_1 - hp1_color, axis=1)
        temp_hp1 = np.ceil( len( np.where(norm_1 == 0)[0] )/255 *100 )
        if temp_hp1 <= self.hp1: self.hp1 = temp_hp1

        temp_2 = np.flip(line[360:615], axis=0)
        norm_2 = np.linalg.norm(temp_2 - hp2_color, axis=1)
        temp_hp2 = np.ceil( len( np.where(norm_2 == 0)[0] )/255 *100 )
        if temp_hp2 <= self.hp2: self.hp2 = temp_hp2
    
    def findStates_detailed(self):
        ## use the YOLO model to get the players' information.
        ## this is the detailed version for reading
        temp_image = cv2.cvtColor(self.game_img, cv2.COLOR_RGB2BGR)
        pred_labels, pred_conf, pred_boxes = readScreenShot4States_detailed(temp_image)
        ## pred_boxes: upper-left horizontal and vertical, lower-right horizontal and vertical

        position_1, position_2 = [-1, -1], [-1, -1]
        label_1, label_2 = -1, -1
        ## if cannot recognize, default as -1

        flag_1, flag_2 = 0, 0
        zoomed = []
        for i in range(len(pred_labels)):
            if (flag_1 == 1) and (flag_2 == 1): break

            if (flag_1 == 0) and (self.states_detailed[pred_labels[i]][0] == "1"):
                this_width, this_height = pred_boxes[i][2]-pred_boxes[i][0], pred_boxes[i][3]-pred_boxes[i][1]
                this_box_size, this_aspect_ratio = this_height * this_width, this_height/this_width
                position_1 = [(pred_boxes[i][0]+pred_boxes[i][2])/2, (pred_boxes[i][1]+pred_boxes[i][3])/2]
                              ## horizontal and vertical positions
                label_1, flag_1 = pred_labels[i], 1

                referenced_aspect_ratio = list(self.sizes_detailed[self.states_detailed[label_1]].keys())
                referenced_aspect_ratio = referenced_aspect_ratio[np.argsort(np.array(referenced_aspect_ratio) - this_aspect_ratio)[0]]
                referenced_size = self.sizes_detailed[self.states_detailed[label_1]][referenced_aspect_ratio]
                referenced_zoom = np.argsort(np.abs(np.array(referenced_size) - this_box_size))[0] ## if 1, zoomed; if 0, not zoomed.
                zoomed.append(referenced_zoom)
                
            if (flag_2 == 0) and (self.states_detailed[pred_labels[i]][0] == "2"):
                this_width, this_height = pred_boxes[i][2]-pred_boxes[i][0], pred_boxes[i][3]-pred_boxes[i][1]
                this_box_size, this_aspect_ratio = this_height * this_width, this_height/this_width
                position_2 = [(pred_boxes[i][0]+pred_boxes[i][2])/2, (pred_boxes[i][1]+pred_boxes[i][3])/2]
                              ## horizontal and vertical positions
                label_2, flag_2 = pred_labels[i], 1

                referenced_aspect_ratio = list(self.sizes_detailed[self.states_detailed[label_2]].keys())
                referenced_aspect_ratio = referenced_aspect_ratio[np.argsort(np.array(referenced_aspect_ratio) - this_aspect_ratio)[0]]
                referenced_size = self.sizes_detailed[self.states_detailed[label_2]][referenced_aspect_ratio]
                referenced_zoom = np.argsort(np.abs(np.array(referenced_size) - this_box_size))[0] ## if 1, zoomed; if 0, not zoomed.
                zoomed.append(referenced_zoom)

        if (flag_1 == 0) and (flag_2 == 1):
            position_1 = position_2
            label_1 = self.states_detailed.index("1_idle")
        elif (flag_2 == 0) and (flag_1 == 1):
            position_2 = position_1
            label_2 = self.states_detailed.index("2_idle")

        self.p1 = position_1 + [label_1]
        self.p2 = position_2 + [label_2]
        self.zoomed = int(1 in zoomed)

    def findStates_detailed_smallWindow(self):
        ## use the YOLO model to get the players' information.
        ## this is the detailed version for reading
        temp_image = cv2.cvtColor(self.game_img, cv2.COLOR_RGB2BGR)
        pred_labels, pred_conf, pred_boxes = readScreenShot4States_detailed(temp_image)
        ## pred_boxes: upper-left horizontal and vertical, lower-right horizontal and vertical

        position_1, position_2 = [-1, -1], [-1, -1]
        label_1, label_2 = -1, -1
        ## if cannot recognize, default as -1

        flag_1, flag_2 = 0, 0
        zoomed = []
        for i in range(len(pred_labels)):
            if (flag_1 == 1) and (flag_2 == 1): break

            if (flag_1 == 0) and (self.states_detailed[pred_labels[i]][0] == "2"):
                this_width, this_height = pred_boxes[i][2]-pred_boxes[i][0], pred_boxes[i][3]-pred_boxes[i][1]
                this_box_size, this_aspect_ratio = this_height * this_width, this_height/this_width
                position_1 = [(pred_boxes[i][0]+pred_boxes[i][2])/2, (pred_boxes[i][1]+pred_boxes[i][3])/2]
                              ## horizontal and vertical positions
                label_1, flag_1 = pred_labels[i], 1

                referenced_aspect_ratio = list(self.sizes_detailed[self.states_detailed[label_1]].keys())
                referenced_aspect_ratio = referenced_aspect_ratio[np.argsort(np.array(referenced_aspect_ratio) - this_aspect_ratio)[0]]
                referenced_size = self.sizes_detailed[self.states_detailed[label_1]][referenced_aspect_ratio]
                referenced_zoom = np.argsort(np.abs(np.array(referenced_size) - this_box_size))[0] ## if 1, zoomed; if 0, not zoomed.
                zoomed.append(referenced_zoom)
                
            if (flag_2 == 0) and (self.states_detailed[pred_labels[i]][0] == "1"):
                this_width, this_height = pred_boxes[i][2]-pred_boxes[i][0], pred_boxes[i][3]-pred_boxes[i][1]
                this_box_size, this_aspect_ratio = this_height * this_width, this_height/this_width
                position_2 = [(pred_boxes[i][0]+pred_boxes[i][2])/2, (pred_boxes[i][1]+pred_boxes[i][3])/2]
                              ## horizontal and vertical positions
                label_2, flag_2 = pred_labels[i], 1

                referenced_aspect_ratio = list(self.sizes_detailed[self.states_detailed[label_2]].keys())
                referenced_aspect_ratio = referenced_aspect_ratio[np.argsort(np.array(referenced_aspect_ratio) - this_aspect_ratio)[0]]
                referenced_size = self.sizes_detailed[self.states_detailed[label_2]][referenced_aspect_ratio]
                referenced_zoom = np.argsort(np.abs(np.array(referenced_size) - this_box_size))[0] ## if 1, zoomed; if 0, not zoomed.
                zoomed.append(referenced_zoom)

        if (flag_1 == 0) and (flag_2 == 1):
            position_1 = position_2
            label_1 = self.states_detailed.index("2_idle")
        elif (flag_2 == 0) and (flag_1 == 1):
            position_2 = position_1
            label_2 = self.states_detailed.index("1_idle")

        self.p1 = position_1 + [label_1]
        self.p2 = position_2 + [label_2]
        self.zoomed = int(1 in zoomed)

    def findHPs(self):
        ## FOR STEAM ##
        ## health points of the two players as reward
        line = np.asarray(self.game_img)[:, :, :][88, :]
        hp1_color, hp2_color = line[449], line[526]

        temp_1 = np.flip(line[147:450], axis=0)
        norm_1 = np.linalg.norm(np.asarray(temp_1, np.float) - np.asarray(hp1_color, np.float) , axis=1)
        temp_hp1 = np.ceil( len( np.where(norm_1 <= 5)[0] )/303 *100 )
        if temp_hp1 <= self.hp1: self.hp1 = temp_hp1

        temp_2 = np.flip(line[526:829], axis=0)
        norm_2 = np.linalg.norm(np.asarray(temp_2, np.float) - np.asarray(hp2_color, np.float), axis=1)
        temp_hp2 = np.ceil( len( np.where(norm_2 <= 5)[0] )/303 *100 )
        if temp_hp2 <= self.hp2: self.hp2 = temp_hp2        

    def findHPs_smallWindow(self):
        ## FOR XIAOHUTAO VERSION ##
        ## health points of the two players as reward
        line = np.asarray(self.game_img)[:, :, :][99, :]
        hp1_color, hp2_color = line[295], line[360]

        temp_1 = np.flip(line[ 40:296], axis=0)
        norm_1 = np.linalg.norm(np.asarray(temp_1, np.float) - np.asarray(hp1_color, np.float), axis=1)
        temp_hp1 = np.ceil( len( np.where(norm_1 <= 5)[0] )/256 *100 )
        if temp_hp1 <= self.hp1: self.hp1 = temp_hp1

        temp_2 = np.flip(line[360:616], axis=0)
        norm_2 = np.linalg.norm(np.asarray(temp_2, np.float) - np.asarray(hp2_color, np.float), axis=1)
        temp_hp2 = np.ceil( len( np.where(norm_2 <= 5)[0] )/256 *100 )
        if temp_hp2 <= self.hp2: self.hp2 = temp_hp2

    def grabGame(self, path=None):
        bbox = win32gui.GetWindowRect(self.handle)
        temp = ImageGrab.grab(bbox)
        if path != None:
            temp.save(path)

        self.game_img = np.array(temp)
        self.findState()
        self.findHPs()

        return np.array([self.label_1] + self.position_1 + [self.box_size_1, self.hp1] + [self.label_2] + self.position_2 + [self.box_size_2, self.hp2])
        ## the information vector

    def endGame(self):
        winMark_h, winMark_w, _ = self.winMark.shape
        img = np.hstack([self.game_img[120:170, 45:180], self.game_img[120:170, 480:615]])
        ## take only the part where it displays
        res = cv2.matchTemplate(img, self.winMark, eval('cv2.TM_CCOEFF'))
        arg = np.vstack(res.flatten().argsort())

        temp = []
        for i in range(8):
            postns = np.where( res == res.flatten()[arg[-(i+1)]] )
            postn = [postns[0][0], postns[1][0]]

            found = img[postn[0]:postn[0]+winMark_h,
                        postn[1]:postn[1]+winMark_w]

            if (len( np.where(found>225)[0] ) > 1024) and (np.linalg.norm(np.tanh((found - self.winMark)/500)) < 13):
                if len(temp) == 0:
                    temp.append(postn)
                else:
                    temp_distances = [np.linalg.norm(np.array(postn) - np.array(j)) for j in temp]
                    temp_distances.sort()

                    if temp_distances[0] <= 20:
                        continue
                    else:
                        temp.append(postn)

        if len(temp) >= 1:
            temp = np.vstack([np.array(j) for j in temp])
            temp = temp[ np.argsort(temp[:, 1]) ]

            n1 = len( np.where(temp[:, 1] < 135)[0] )
            n2 = len( np.where(temp[:, 1] > 135)[0] )

            if n1 == 2:
                return "1_win"
            elif n2 == 2:
                return "2_win"
            elif n1 + n2 == 1:
                return "midgame"
            elif (n1 == 1) and (n2 == 1):
                return "draw"

        return ""

    def activateGame(self):
        ## Bring to top the game window
        
        shell = win32com.client.Dispatch("WScript.Shell")
        shell.SendKeys('%')
        win32gui.SetForegroundWindow(self.handle)
        # if reactivate:
            ## This is abandoned as the restoring process is translating the window.
            # win32gui.ShowWindow(self.handle, win32con.SW_RESTORE)
            # time.sleep(0.3)

        time.sleep(0.5)
        press("esc")
        time.sleep(0.3)
        press("right")
        time.sleep(0.3)
        press("enter")

    def reloadFromSave(self):
        keyDown("alt")
        press("g")
        keyUp("alt")
        time.sleep(0.75)
        press("enter")
        time.sleep(0.75)
        press("u")
        time.sleep(0.75)
        press("u")
        time.sleep(2.75)
        press("s")
        time.sleep(0.75)
        press("u")
        time.sleep(1)
        press("u")
        time.sleep(5)

    def deactivateGame(self):
        win32gui.PostMessage(self.handle, win32con.WM_CLOSE, 0, 0)

    def minimizeGame(self):
        ## minimize the window for another re-activation.
        win32gui.ShowWindow(self.handle, win32con.SW_SHOWMINIMIZED)

    def pauseGame(self):
        press("esc")

    def resumeGame(self):
        press("right")
        time.sleep(0.2)
        press("enter")
