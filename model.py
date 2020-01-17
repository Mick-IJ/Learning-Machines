import numpy as np
import random


class Model:
    def __init__(self, q_table):
        self.q_table = q_table

    def predict(self, sensors):
        state = self.get_state(sensors)
        action = random.sample(np.argwhere(self.q_table[state] == np.amax(self.q_table[state]))
                               .flatten().tolist(), 1)[0]

        if action == 0:
            left, right, duration = -25, 25, 300
        elif action == 1:
            left, right, duration = 25, -25, 300
        else:  # action == 2:
            left, right, duration = 25, 25, 300

        return left, right, duration

    def get_state(self, sensors):
        sensors = np.log(np.array(sensors)) / 10  # scale sensors
        sensors = np.where(sensors == -np.inf, 0, sensors)  # remove the infinite
        sensors = [1 if sensor > 0.5 else 0 for sensor in sensors]  # binarize sensors

        r = max(sensors[3:6])
        l = max(sensors[6:8])

        return int(str(l)+str(r), 2)

    OPTIMAL_QTABLE = (np.array([
                np.array([0, 0, 1]),
                np.array([1, 0, 0]),
                np.array([0, 1, 0]),
                np.array([0, 1, 0])]))  # Found from the evolution
