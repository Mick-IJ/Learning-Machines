from __future__ import print_function
import time
import numpy as np
import robobo
import cv2
import sys
import signal
import prey
import matplotlib.pyplot as plt
import random
import pickle


class Environment:
    def __init__(self):
        signal.signal(signal.SIGINT, self.terminate_program)
        self.hunter = robobo.SimulationRobobo(number='').connect(address='127.0.0.1', port=19999)
        self.prey = robobo.SimulationRobobo(number='#0').connect(address='127.0.0.1', port=19998)

        self.time_since_obj = {'Hunter': [0, 0], 'Prey': [0, 0]}

    def terminate_program(self, signal_number, frame):
        print("Ctrl-C received, terminating program")
        sys.exit(1)

    def select_play(self, hunters, preys, best_hunter=False, best_prey=False, n_its=100):
        if best_hunter and best_prey:
            self.play(hunters.get_best(), preys.get_best())
        elif best_hunter:
            for prey in preys.individuals:
                self.play(hunters.get_best(), prey)
        elif best_prey:
            for hunter in hunters.individuals:
                self.play(hunter, preys.get_best)
        else:  # not best_hunter and not best_prey
            for i in range(preys.size):
                self.play(hunters.individuals[i], preys.individuals[i])

        return hunters, preys

    def play(self, hunter, prey, its=100):
        self.hunter.play_simulation()
        self.prey.play_simulation()
        self.hunter.set_phone_tilt(0.6, 10)

        q_hunter = hunter.q_table
        q_prey = prey.q_table

        for i in range(its):
            states = self.get_states()
            state_hunter, crash_hunter, caught_hunter = states[0]
            state_prey, crash_prey, caught_prey = states[1]

            if caught_hunter or caught_prey:
                break
            if crash_hunter:
                hunter.fitness -= 50
            if crash_prey:
                prey.fitness -= 50

            l_hunter, r_hunter, d_hunter = self.get_move(q_hunter, state_hunter)
            l_prey, r_prey, d_prey = self.get_move(q_prey, state_prey)

            self.hunter.move(l_hunter, r_hunter, d_hunter)
            self.prey.move(l_prey, r_prey, d_prey)

            hunter.fitness -= 1
            prey.fitness += 1

        print(hunter.fitness, prey.fitness)

    def get_move(self, q_table, state):
        move = np.argmax(q_table[state])

        if move == 0:
            left, right, duration = -25, 25, 300
        elif move == 1:
            left, right, duration = 25, -25, 300
        else:  # move == 2:
            left, right, duration = 25, 25, 300

        return left, right, duration

    def get_states(self):
        states = []
        for name, rob in [('Hunter', self.hunter), ('Prey', self.prey)]:
            sensors = np.log(np.array(rob.read_irs())) / 10
            sensors = np.where(sensors == -np.inf, 0, sensors)  # remove the infinite
            sensors = (sensors - -0.65) / 0.65
            n_sensors = [(1 if sensor < 0.7 else 0) for sensor in sensors]

            r, l = max(n_sensors[3:6]), max(n_sensors[6:8])
            o = self.detect_object(rob, name)
            t = 1 if min(self.time_since_obj[name]) < 10 else 0
            l_recent = 1 if self.time_since_obj[name][0] <= self.time_since_obj[name][1] else 0

            crash = True if not o and min(sensors) < 0.3 else False
            caught = True if o and min(sensors) < 0.3 else False

            states.append((int(str(l_recent)+str(t)+str(o)+str(l)+str(r), 2), crash, caught))
        return states

    def detect_object(self, rob, name):
        input_ = rob.get_image_front()
        inputL, inputR = input_[:, :62, :], input_[:, 62:, :]

        maskL = cv2.inRange(inputL, (0, 0, 100), (140, 140, 255))
        maskR = cv2.inRange(inputR, (0, 0, 100), (140, 140, 255))

        contoursL, _ = cv2.findContours(maskL, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contoursR, _ = cv2.findContours(maskR, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if contoursL or contoursR:
            if contoursL and contoursR:
                self.time_since_obj[name][0] = 0
                self.time_since_obj[name][1] = 0
            elif contoursR:
                self.time_since_obj[name][1] = 0
                self.time_since_obj[name][0] += 1
            else:  # contoursL
                self.time_since_obj[name][0] = 0
                self.time_since_obj[name][1] += 1
            return 1
        else:
            self.time_since_obj[name][0] += 1
            self.time_since_obj[name][1] += 1
            return 0


class Individual:
    dom_u = 1
    dom_l = 0

    def __init__(self):
        self.age = 0
        self.q_table = list()
        self.fitness = 0
        self.time = None
        self.children = 0
        self.parents = None
        self.n_states = 32  # 2bits sensor + 1bit cam detected
        self.n_actions = 3
        self.fit_hist = []

    def set_q_table(self, q_table=None):
        if q_table is None:
            q_table = list()
            for state in range(self.n_states):
                row = np.zeros((1, self.n_actions))[0]
                index_ = random.sample([i for i in range(self.n_actions)], 1)[0]
                row[index_] = 1
                q_table.append(row)
            self.q_table = np.array(q_table)
        else:
            self.q_table = q_table

    def birthday(self):
        self.age += 1

    def mutate(self, mutation_rate):
        for i, row in enumerate(self.q_table):
            if np.random.random() <= mutation_rate:
                np.random.shuffle(self.q_table[i])


class Population:
    def __init__(self, size=5):
        self.individuals = list()
        self.size = size
        self.generation = 1
        self.mutation_rate = 0.1
        self.mean_age = None
        self.mean_children = None
        self.mean_fit = None
        self.max_fit = None
        self.worst_fit = None
        self.var = None
        self.mean_fit_history = list()
        self.max_fit_history = list()
        self.worst_fit_history = list()
        self.var_history = list()
        self.best_individual = None

    def append(self, individual):
        self.individuals.append(individual)

    def extend(self, population):
        self.individuals.extend(population)
        self.update_stats()

    def kill(self, individual):
        self.individuals.remove(individual)

    def get_best(self):
        for individual in self.individuals:
            if self.best_individual:
                if individual.fitness > self.best_individual.fitness:
                    self.best_individual = individual
            else:
                self.best_individual = individual

        return self.best_individual

    def update_stats(self):
        population_fit = [i.fitness for i in self.individuals]
        self.var = np.var(population_fit)
        self.mean_fit = np.mean(population_fit)
        self.max_fit = np.max(population_fit)
        self.worst_fit = np.min(population_fit)
        self.mean_age = np.mean([i.age for i in self.individuals])
        self.mean_children = np.mean([i.children for i in self.individuals])
        self.best_individual = self.get_best()
        #pickle.dump(self, open('pop7.txt', 'wb'))

    def display_population(self):
        for i, individual in enumerate(self.individuals):
            if individual.parents is not None:
                parent1, parent2 = individual.parents
                mean_parents = (parent1.fitness + parent2.fitness) / 2
            else:
                mean_parents = 0
            print(i, ': fitness =', round(individual.fitness, 4), 'age =', individual.age,
                  'children =', individual.children, 'parent_fit =', round(mean_parents, 2))

        print('Mean fitness:', round(self.mean_fit, 4), 'Mean age:', round(self.mean_age, 2),
              'Mean children =', round(self.mean_children, 2), '\n')

    def initialize(self):
        for i in range(self.size):
            individual = Individual()
            individual.set_q_table()
            individual.birthday()
            self.individuals.append(individual)

    def select_parents(self, n, type_='random'):
        if type_ == 'random':
            return random.sample(self.individuals, n)
        elif type_ == 'fit':
            pop_fitness = [(individual.fitness + self.worst_fit) for individual in self.individuals]
            probabilities = [(fit / sum(pop_fitness)) for fit in pop_fitness]
            return np.random.choice(self.individuals, size=n, replace=False, p=probabilities)
        elif type_ == 'rank':
            pop_fitness = [(individual.fitness + self.worst_fit) for individual in self.individuals]
            ranks = [sorted(pop_fitness).index(ind) + 1 for ind in pop_fitness]
            probabilities = [rank / sum(ranks) for rank in ranks]
            return np.random.choice(self.individuals, size=n, replace=False, p=probabilities)

    def trim(self, type_='fit'):
        size = self.size

        if type_ == 'random':
            self.individuals = random.sample(self.individuals, size)
        elif type_ == 'fit':
            pop_fitness = [(individual.fitness + self.worst_fit) for individual in self.individuals]
            probabilities = [(fit / sum(pop_fitness)) for fit in pop_fitness]
            self.individuals = list(np.random.choice(self.individuals, size=size, replace=False, p=probabilities))
        elif type_ == 'rank':
            pop_fitness = [(individual.fitness + self.worst_fit) for individual in self.individuals]
            ranks = [(sorted(pop_fitness).index(ind) + 1) for ind in pop_fitness]
            probabilities = [rank / sum(ranks) for rank in ranks]
            self.individuals = list(np.random.choice(self.individuals, size=size, replace=False, p=probabilities))

        self.update_stats()

    def sex(self, selection_type='fit'):
        parent1, parent2 = self.select_parents(2, type_=selection_type)
        child1 = np.zeros((parent1.n_states, parent1.n_actions))
        child2 = np.zeros((parent1.n_states, parent1.n_actions))

        for i, row in enumerate(child1):
            if np.random.random() <= .5:
                child1[i] = parent1.q_table[i]
                child2[i] = parent1.q_table[i]
            else:
                child1[i] = parent2.q_table[i]
                child2[i] = parent2.q_table[i]

        children = [child1, child2]

        for child_q_table in children:
            child = Individual()
            child.set_q_table(child_q_table)
            child.mutate(mutation_rate=self.mutation_rate)
            child.parents = (parent1, parent2)
            self.individuals.append(child)

            parent1.children += 1
            parent2.children += 1

    def mutation_rate_change(self, type_='exp'):
        if type_ == 'linear':
            self.mutation_rate -= 0.01
        elif type_ == 'exp':
            self.mutation_rate *= 0.999
        elif type_ == 'log':
            self.mutation_rate = np.log(self.mutation_rate)

    def next_generation(self):
        self.generation += 1
        for individual in self.individuals:
            individual.birthday()
        self.mean_fit_history.append(self.mean_fit)
        self.max_fit_history.append(self.max_fit)
        self.worst_fit_history.append(self.worst_fit)
        self.var_history.append(self.var)

    def plot_generations(self):
        plt.clf()
        plt.style.use('seaborn')
        plt.figure(figsize=(5, 5))

        x = np.linspace(0, self.generation, self.generation)
        y, error = np.array(self.mean_fit_history), np.array(self.var_history)

        plt.plot(x, y, 'k', label='Mean', color='#CC4F1B')
        plt.fill_between(x, y - error, y + error,
                         alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')

        plt.plot(self.max_fit_history, label="Best", color='#88CC1B')
        plt.plot(self.worst_fit_history, label='Worst', color='#CC1B1B')
        plt.ylim((0, 400))
        plt.ylabel('Fitness')
        plt.xlabel('Generation')
        plt.legend()
        plt.show()


def main(size=5, generations=5, children_per_gen=5):
    hunters, preys = Population(size=size), Population(size=size)
    hunters.initialize()

    ind = Individual()
    ind.set_q_table(np.array([# Left, Right, Forward // # L_recent, T, O, L, RF
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
        ]))
    ind.fitness = 100
    hunters.append(ind)

    preys.initialize()

    ENV = Environment()

    ENV.select_play(hunters, preys, best_hunter=True, best_prey=False)

main(size=2)
