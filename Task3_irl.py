from __future__ import print_function
import time
import numpy as np
import robobo
import cv2
import sys
import signal
import prey


class Environment:
    def __init__(self):
        signal.signal(signal.SIGINT, self.terminate_program)
        self.rob = robobo.SimulationRobobo(number='').connect(address='127.0.0.1', port=19999)                          ##CHECK
        self.time_since_obj = [0, 0]

    def terminate_program(self, signal_number, frame):
        print("Ctrl-C received, terminating program")
        sys.exit(1)

    def play(self, q_table, its=100):
        self.rob.play_simulation()
        self.rob.set_phone_tilt(125, 10)                                                                                ##CHECK

        for i in range(its):
            state, close, o = self.get_states()

            if close and o:
                break
            if close and not o:
                break

            left, right, duration = self.get_move(q_table, state)

            self.rob.move(left, right, duration)

        self.rob.stop_world()
        self.rob.wait_for_stop()

    def get_move(self, q_table, state):
        move = np.argmax(q_table[state])

        if move == 0:
            left, right, duration = -25, 25, 300
        elif move == 1:
            left, right, duration = 25, -25, 300
        else:  # move == 2:
            left, right, duration = 25, 25, 300

            left += np.random.randint(0, 10, 1)[0]
            right += np.random.randint(0, 10, 1)[0]

        return left, right, duration

    def get_states(self):
        sensors = np.log(np.array(self.rob.read_irs())) / 10
        sensors = np.where(sensors == -np.inf, 0, sensors)  # remove the infinite
        n_sensors = [(1 if sensor > 0.3 else 0) for sensor in sensors]                                                  ##CHECK

        r, l = max(n_sensors[3:6]), max(n_sensors[6:8])
        o = self.detect_object()
        t = 1 if min(self.time_since_obj) < 16 else 0                                                                   #CHECK
        l_recent = 1 if self.time_since_obj[0] <= self.time_since_obj[1] else 0

        return int(str(l_recent)+str(t)+str(o)+str(l)+str(r), 2), min(sensors) < 0.3, o                                 ##CHECK

    def detect_object(self):
        input_ = self.rob.get_image_front()
        inputL, inputR = input_[:600, 140:240, :], input_[:600, 240:360, :]                                             ##CHECK

        maskL = cv2.inRange(inputL, (0, 0, 100), (140, 140, 255))
        maskR = cv2.inRange(inputR, (0, 0, 100), (140, 140, 255))

        contoursL, _ = cv2.findContours(maskL, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contoursR, _ = cv2.findContours(maskR, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if contoursL or contoursR:
            if contoursL and contoursR:
                self.time_since_obj[0] = 0
                self.time_since_obj[1] = 0
            elif contoursR:
                self.time_since_obj[1] = 0
                self.time_since_obj[0] += 1
            else:  # contoursL
                self.time_since_obj[0] = 0
                self.time_since_obj[1] += 1
            return 1
        else:
            self.time_since_obj[0] += 1
            self.time_since_obj[1] += 1
            return 0


def play_best():
    q_hunter = np.array([# Left, Right, Forward // # L_recent, T, O, L, RF
            np.array([0, 0, 1]),  # 00000
            np.array([1, 0, 0]),  # 00001
            np.array([0, 1, 0]),  # 00010
            np.array([1, 0, 0]),  # 00011
            np.array([0, 0, 1]),  # 00100
            np.array([0, 0, 1]),  # 00101
            np.array([0, 0, 1]),  # 00110
            np.array([0, 0, 1]),  # 00111

            np.array([0, 1, 0]),  # 01000
            np.array([0, 1, 0]),  # 01001
            np.array([0, 1, 0]),  # 01010
            np.array([0, 1, 0]),  # 01011
            np.array([0, 0, 1]),  # 01100
            np.array([0, 0, 1]),  # 01101
            np.array([0, 0, 1]),  # 01110
            np.array([0, 0, 1]),  # 01111

            np.array([0, 0, 1]),  # 10000
            np.array([1, 0, 0]),  # 10001
            np.array([0, 1, 0]),  # 10010
            np.array([1, 0, 0]),  # 10011
            np.array([0, 0, 1]),  # 10100
            np.array([0, 0, 1]),  # 10101
            np.array([0, 0, 1]),  # 10110
            np.array([0, 0, 1]),  # 10111

            np.array([1, 0, 0]),  # 11000
            np.array([1, 0, 0]),  # 11001
            np.array([1, 0, 0]),  # 11010
            np.array([1, 0, 0]),  # 11011
            np.array([0, 0, 1]),  # 11100
            np.array([0, 0, 1]),  # 11101
            np.array([0, 0, 1]),  # 11110
            np.array([0, 0, 1]),  # 11111
        ])

    ENV = Environment()
    for i in range(10):
        ENV.play(q_hunter, its=2000)


play_best()

q_prey = np.array(  [# Left, Right, Forward // # L_recent, T, O, L, RF
    np.array([0, 1, 0]),  # 00000
    np.array([1, 0, 0]),  # 00001
    np.array([0, 1, 0]),  # 00010
    np.array([1, 0, 0]),  # 00011
    np.array([0, 0, 1]),  # 00100
    np.array([1, 0, 0]),  # 00101
    np.array([0, 1, 0]),  # 00110
    np.array([1, 0, 0]),  # 00111

    np.array([0, 0, 1]),  # 01000
    np.array([1, 0, 0]),  # 01001
    np.array([0, 1, 0]),  # 01010
    np.array([1, 0, 0]),  # 01011
    np.array([0, 0, 1]),  # 01100
    np.array([1, 0, 0]),  # 01101
    np.array([0, 1, 0]),  # 01110
    np.array([1, 0, 0]),  # 01111

    np.array([1, 0, 0]),  # 10000
    np.array([1, 0, 0]),  # 10001
    np.array([0, 1, 0]),  # 10010
    np.array([1, 0, 0]),  # 10011
    np.array([0, 0, 1]),  # 10100
    np.array([1, 0, 0]),  # 10101
    np.array([0, 1, 0]),  # 10110
    np.array([1, 0, 0]),  # 10111

    np.array([0, 0, 1]),  # 11000
    np.array([1, 0, 0]),  # 11001
    np.array([0, 1, 0]),  # 11010
    np.array([1, 0, 0]),  # 11011
    np.array([0, 0, 1]),  # 11100
    np.array([1, 0, 0]),  # 11101
    np.array([0, 1, 0]),  # 11110
    np.array([1, 0, 0]),  # 11111
])

