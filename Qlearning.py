from __future__ import print_function
import time
import numpy as np
import robobo
import cv2
import sys
import signal
import prey
import matplotlib.pyplot as plt
from random import sample
import random


def terminate_program(self, signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)


signal.signal(signal.SIGINT, terminate_program)
rob = robobo.SimulationRobobo(number='#0').connect(address='127.0.0.1', port=19997)
rob.play_simulation()


def get_state():
    global rob
    sensors = np.log(np.array(rob.read_irs())) / 10
    sensors = np.where(sensors == -np.inf, 0, sensors)  # remove the infinite
    sensors = (sensors - -0.65) / 0.65

    n_sensors = []
    for sensor in sensors:
        if sensor < 0.6:
            r = 1
        else:
            r = 0
        n_sensors.append(r)

    b = max(n_sensors[:3])
    r = max(n_sensors[3:6])
    l = max(n_sensors[5:8])

    return int(str(b)+str(r)+str(l), 2)


def get_reward(move):
    global rob
    sensors = np.log(np.array(rob.read_irs())) / 10
    sensors = np.where(sensors == -np.inf, 0, sensors)  # remove the infinite
    sensors = (sensors - -0.65) / (0 - -0.65)  # scale between 0 and 1

    if min(sensors) > 0.4:
        if move == 0 or move == 1:
            return -5
        elif move == 2:
            return -1
        else:  # move == 3:
            return -2
    else:
        return -20


def get_move(reward_table, state, epsilon):
    move = random.sample(np.argwhere(reward_table[state] == np.amax(reward_table[state]))
                         .flatten().tolist(), 1)[0]
    if epsilon and random.random() < epsilon:
        options = [0, 1, 2, 3]
        options.remove(move)
        move = random.sample(options, 1)[0]

    return move


def main(iterations=100, alpha=0.6, gamma=0.9, method='SARSA', epsilon=0.2):
    '''
    iterations: number of times the game is played
    alpha: learning rate
    gamma: discounting rate
    method: SARSA or QLearning
    epsilon: epsilon in the epsilon greedy policy
    rows: number of rows (<10)
    '''
    global rob
    reward_table = np.zeros((16, 4))  # L, R, F, B

    for i in range(iterations):
        state = get_state()
        old_state = state

        # move from the old state
        move = get_move(reward_table, old_state, epsilon)
        if move == 0:
            left, right = -20, 20
        elif move == 1:
            left, right = 20, -20
        elif move == 2:
            left, right = 25, 25
        else:  # move == 3
            left, right = -25, -25

        rob.move(left, right, 200)
        reward = get_reward(move)
        current_state = get_state()

        # move from the current state
        next_move = get_move(reward_table, current_state, epsilon)

        if method == 'SARSA':
            reward_table[old_state][move] = round(reward_table[old_state][move] +
                                                  alpha * (reward + gamma * reward_table[current_state][next_move]
                                                           - reward_table[old_state][move]), 4)
        else:  # method == 'QLearning'
            reward_table[old_state][move] = round(reward_table[old_state][move] +
                                                  alpha * (reward + gamma * reward_table[current_state].max()
                                                           - reward_table[old_state][move]), 4)

    rob.disconnect()

    return reward_table


table = main(iterations=500)
print(table)