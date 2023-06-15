import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from mcpi.minecraft import Minecraft
import pynput
from pynput.keyboard import Key, Controller
import Minecraft_Manipulate as MM
from HistCorr import HistCorr
import time
import os

mc = Minecraft.create()



# settings
num_states = 144  # number of states, simi: leftpre + leftblock(6) * simi: rightpre + rightblock(6) * position(4)
num_actions = 8
alpha = 0.5
gamma = 0.8
target_position = [-722, -199, 32]
radius = 100
epsilon = 1       # Initial exploration rate
decay_rate = 0.97  # Decay rate for epsilon
minimum_distance = 0.6
max_train = 1500
max_test = 200

reward_factor = 1.05
transmission_factor = 15
plot_factor = 50

base_path = 'C:/Users/12866/Desktop/minecraft_project/training_results/training_after_1500_3'




def update_prelist(prelist, image, size):
    if len(prelist) < size:
        prelist.append(image)
    else:
        prelist.pop(0)
        prelist.append(image)

    return prelist



def update_blocklist(blocklist, image, size):
    if len(blocklist) < size:
        blocklist.append(image)
    else:
        blocklist.pop(0)
        blocklist.append(image)

    return blocklist



def calculate_state(corr):
    state = 0
    if corr < 0.25:
        state = 1
    elif corr < 0.75:
        state = 2
    else:
        state = 3

    return state



def get_leftsimi_state(left_cur_image, left_pre_list, left_block_list):
    left_pre_corr = HistCorr(left_cur_image, left_pre_list).get_corr()
    left_block_corr = HistCorr(left_cur_image, left_block_list).get_corr()

    return calculate_state(left_pre_corr) + calculate_state(left_block_corr)



def get_rightsimi_state(right_cur_image, right_pre_list, right_block_list):
    right_pre_corr = HistCorr(right_cur_image, right_pre_list).get_corr()
    right_block_corr = HistCorr(right_cur_image, right_block_list).get_corr()

    return calculate_state(right_pre_corr) + calculate_state(right_block_corr)



def detect_position_state(target_position, cur_position):
    target_x = target_position[0]
    target_z = target_position[1]

    # cur_position = mc.player.getTilePos()
    cur_x = cur_position.x
    cur_z = cur_position.z

    state = 0
    # target is at Southeast
    if cur_x < target_x and cur_z < target_z:
        # target is at Southeast
        state = 1
    elif cur_x > target_x and cur_z < target_z:
        # target is at Southwest
        state = 2
    elif cur_x < target_x and cur_z > target_z:
        # target is at Northeast
        state = 3
    else:
        # target is at Northwest
        state = 4

    return state



def initialize_qtable():
    qtable = np.random.uniform(low=-0.5, high=0.5, size=(num_states, num_actions))
    # qtable = np.zeros((num_states, num_actions))
    return qtable



def movement(qtable, s):
    MM.init_direction()
    global epsilon

    # higher random chance at the first several iterations
    if np.random.rand() < epsilon:
        a = np.random.choice(np.argsort(qtable[s])[-3:])
    else:
        a = np.argmax(qtable[s])
    epsilon *= decay_rate
    
    if a == 0:
        MM.move2()
    if a == 1:
        MM.turn_back()
        MM.move2()
    if a == 2:
        MM.turn_left()
        MM.move2()
    if a == 3:
        MM.turn_right()
        MM.move2()
    if a == 4:
        MM.move1()
    if a == 5:
        MM.turn_back()
        MM.move1()
    if a == 6:
        MM.turn_left()
        MM.move1()
    if a == 7:
        MM.turn_right()
        MM.move1()
    if a < 4:
        return a, 5
    else:
        return a, 2
    


def get_reward(true_dist, expected_dist, position):
    reward = true_dist/expected_dist * 1

    current_direction = mc.player.getRotation()
    target_state = detect_position_state(target_position, position)

    if 70 <= current_direction <= 110 and (target_state == 2 or target_state == 4):
        reward *= reward_factor
    elif 160 <= current_direction <= 200 and (target_state == 3 or target_state == 4):
        reward *= reward_factor
    elif 250 <= current_direction <= 290 and (target_state == 1 or target_state == 3):
        reward *= reward_factor
    elif (current_direction <= 20 or 340 <= current_direction <= 360) and (target_state == 1 or target_state == 2):
        reward *= reward_factor


    if reward < minimum_distance:
        reward = -10

    return reward



def split_image(image):
    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Crop the left half of the left image
    left_half = image[0:height, 0:int(width/2)]
    # Crop the right half of the right image
    right_half = image[0:height, int(width/2):width]

    return left_half, right_half



def transmission_random():
    # calculate the boundaries of the square
    x_min = target_position[0] - radius
    x_max = target_position[0] + radius
    z_min = target_position[1] - radius
    z_max = target_position[1] + radius

    y = target_position[2]


    while True:
        random_position = [random.uniform(x_min, x_max), random.uniform(z_min, z_max)]
        if detect_standing_block(random_position, y):
            break
        
    mc.player.setTilePos(random_position[0], y, random_position[1])



def detect_standing_block(position, y):

    if ((mc.getBlock(position[0], y-1, position[1])) not in [0, 9, 11] and mc.getBlock((position[0], y, position[1])) == 0
        and mc.getBlock((position[0], y+1, position[1])) == 0):
        return True
    return False



def store_text(path, text_name, text):
    np.savetxt(path + '/' + text_name, text)



def store_images(path, image_list):
    for i, image in enumerate(image_list):
        filename = f"{path}/image_{i}.png"
        img = Image.fromarray(image)
        img.save(filename, "PNG")



def load_text(path):
    qtable = np.loadtxt(path)
    return qtable



def load_images(path):
    image_list = []

    # Iterate through file names and load images
    for i in range(100):
        filename = os.path.join(path, f"image_{i}.png")
        if not os.path.exists(filename):
            break
        img = Image.open(filename)
        image_list.append(np.array(img))

    return image_list



def qlearning():

    # initialize direction
    time.sleep(5)
    MM.init_direction()

    # initialize image
    useless_image, block_image = MM.resize_image(f"C:/Users/12866/AppData/Roaming/.minecraft/screenshots/sample_pre.png")


    cur_image = MM.take_photo()
    left_cur_image, right_cur_image = split_image(cur_image)
    new_position = mc.player.getTilePos()

    left_pre_list = []
    right_pre_list = []
    left_pre_list = update_prelist(left_pre_list, left_cur_image, 15)
    right_pre_list = update_prelist(right_pre_list, right_cur_image, 15)


    left_block_list = []
    right_block_list = []
    left_block_image, right_block_image = split_image(block_image)
    left_block_list = update_blocklist(left_block_list, left_block_image, 100)
    right_block_list = update_blocklist(right_block_list, right_block_image, 100)

    norms = []  # list to store the norms between old and new qtables


    # initilize qtable and state
    qtable = initialize_qtable()
    old_qtable = np.copy(qtable)

    s = random.randint(0, num_states-1)

    hit_count_dp = []


    for i in range(max_train):
        if (i >= plot_factor and i % plot_factor == 0):
            norm = np.linalg.norm(qtable - old_qtable)
            old_qtable = np.copy(qtable)
            norms.append(norm)


        # old position get state
        old_position = new_position

        
        if (i+1) % transmission_factor == 0:
            transmission_random()
                 
    
        # get action and do the movement
        a, expected_dist = movement(qtable, s)
        new_position = mc.player.getTilePos()
        true_dist = np.linalg.norm(np.array([new_position.x, new_position.z])-np.array([old_position.x, old_position.z]))

        # hit the wall
        if true_dist/expected_dist * 1 < minimum_distance:
            # block_image = MM.take_photo()
            # left_block_image, right_block_image = split_image(cur_image)
            left_block_list = update_blocklist(left_block_list, left_cur_image, 100)
            right_block_list = update_blocklist(right_block_list, right_cur_image, 100)

            hit_count_dp.append(1)
        else:
            # update pre image list
            left_pre_list = update_prelist(left_pre_list, left_cur_image, 15)
            right_pre_list = update_prelist(right_pre_list, right_cur_image, 15)
            hit_count_dp.append(0)

        # get reward
        reward = get_reward(true_dist, expected_dist, old_position)
        
        cur_image = MM.take_photo()
        left_cur_image, right_cur_image = split_image(cur_image)
        
        
        s_prime = (get_leftsimi_state(left_cur_image, left_pre_list, left_block_list)
            * get_rightsimi_state(right_cur_image, right_pre_list, right_block_list)
            * detect_position_state(target_position, new_position)) - 1
        
        ## update q table
        # Q(s,a) = Q(s,a) + alpha(r + gamma * max(Q(s', a'))-Q(s,a))

        qtable[s, a] = qtable[s, a] + alpha * (reward + gamma * np.max(qtable[s_prime]) - qtable[s, a])
        
        s = s_prime



    store_text(base_path, 'qtable.txt', qtable)
    store_images(base_path + '/left_pre_list', left_pre_list)
    store_images(base_path + '/right_pre_list', right_pre_list)
    store_images(base_path + '/left_block_list', left_block_list)
    store_images(base_path + '/right_block_list', right_block_list)

    # plot the norms against the iteration numbers
    # range(start, stop, step)
    store_text(base_path, 'norm.txt', norms)
    plt.plot(np.arange(0, max_train + plot_factor, plot_factor)[:len(norms)], norms)
    plt.xlabel('Iteration')
    plt.ylabel('Norm between old and new qtables')
    plt.savefig(base_path + '/norm_plot.png')
    plt.show()
    


    # define the intervals
    interval_size = 100
    index = 0
    hit_counts = []
    while (index+interval_size) <= len(hit_count_dp):
        hit_counts.append(sum(hit_count_dp[index : index + interval_size]))
        index += 10
    
    x_labels = []
    for i in range(len(hit_counts)):
        x_labels.append(str(i*10) + '-' + str(100 + 10 * i))
        
    # create the plot
    plt.figure()
    plt.plot(x_labels, hit_counts)
    plt.xlabel('Interval')
    plt.ylabel('Hitting Rate')
    plt.savefig(base_path + '/hitting_rate_plot.png')
    plt.show()







def navigation():
    # initialize direction
    time.sleep(5)
    MM.init_direction()


    cur_image = MM.take_photo()
    left_cur_image, right_cur_image = split_image(cur_image)
    new_position = mc.player.getTilePos()


    global epsilon
    epsilon = 0


    qtable = load_text(f"{base_path}/qtable.txt")
    left_pre_list = load_images(f"{base_path}/right_pre_list")
    right_pre_list = load_images(f"{base_path}/right_pre_list")
    left_block_list = load_images(f"{base_path}/left_block_list")
    right_block_list = load_images(f"{base_path}/right_block_list")

    left_pre_list = update_prelist(left_pre_list, left_cur_image, 20)
    right_pre_list = update_prelist(right_pre_list, right_cur_image, 20)



    s = (get_leftsimi_state(left_cur_image, left_pre_list, left_block_list)
            * get_rightsimi_state(right_cur_image, right_pre_list, right_block_list)
            * detect_position_state(target_position, new_position)) - 1
    
    hit_count_dp = []
    
    # norm(curr_pos-target_pos)
    norms_list = []
    x_labels = []


    for i in range(max_test):
        # old position get state
        old_position = new_position

        if (np.abs(old_position.x - target_position[0]) <= 10) and (np.abs(old_position.z - target_position[1]) <= 10):
            print("success")
            break

        
        # get action and do the movement
        a, expected_dist = movement(qtable, s)
        new_position = mc.player.getTilePos()
        true_dist = np.linalg.norm(np.array([new_position.x, new_position.z])-np.array([old_position.x, old_position.z]))

        # hit the wall
        if true_dist/expected_dist * 1 < minimum_distance:
            # block_image = MM.take_photo()
            # left_block_image, right_block_image = split_image(cur_image)
            left_block_list = update_blocklist(left_block_list, left_cur_image, 100)
            right_block_list = update_blocklist(right_block_list, right_cur_image, 100)

            hit_count_dp.append(1)

        else:
            # update pre image list
            left_pre_list = update_prelist(left_pre_list, left_cur_image, 15)
            right_pre_list = update_prelist(right_pre_list, right_cur_image, 15)

            hit_count_dp.append(0)


        old = np.array([old_position.x, old_position.z, old_position.y])
        target = np.array([target_position[0], target_position[1], target_position[2]])
        distance = np.linalg.norm(old - target)
        norms_list.append(distance)
        x_labels.append(i)



        
        cur_image = MM.take_photo()
        left_cur_image, right_cur_image = split_image(cur_image)
        
        
        s_prime = (get_leftsimi_state(left_cur_image, left_pre_list, left_block_list)
            * get_rightsimi_state(right_cur_image, right_pre_list, right_block_list)
            * detect_position_state(target_position, new_position)) - 1
        
        
        s = s_prime
    
    # create the plot
    plt.figure()
    plt.plot(x_labels, norms_list)
    plt.xlabel('Interations')
    plt.ylabel('Norm of cur_pos minus tar_pos')
    plt.savefig(base_path + '/distance_norm_plot.png')
    plt.show()



    # define the intervals
    interval_size = 100
    index = 0
    hit_counts = []
    while (index+interval_size) <= len(hit_count_dp):
        hit_counts.append(sum(hit_count_dp[index : index + interval_size]))
        index += 10
    
    x_labels = []
    for i in range(len(hit_counts)):
        x_labels.append(str(i*10) + '-' + str(100 + 10 * i))
        
    # create the plot
    plt.figure()
    plt.plot(x_labels, hit_counts)
    plt.xlabel('Interval')
    plt.ylabel('Hitting Rate')
    plt.savefig(base_path + '/test_hitting_rate_plot.png')
    plt.show()





def main():
    # path = f"C:/Users/12866/AppData/Roaming/.minecraft/screenshots/2.png"
    # image, resized_image = MM.resize_image(path)
    # left, right = split_image(resized_image)
    # MM.show_picture(left, right)
    # time.sleep(3)
    # for i in range(3):
    #     photo = MM.take_photo()
    #     left, right = split_image(photo)
    #     MM.show_picture(left, right)

    # qlearning()
    navigation()
    


    # time.sleep(5)
    # photo = MM.take_photo()
    # MM.show_picture(photo, photo)
    # mc.player.setTilePos(-700, 32, -140)
    # navigation(qtable, left_pre_list, right_pre_list, left_block_list, right_block_list)


    

    




# The only place run code
main()

    
