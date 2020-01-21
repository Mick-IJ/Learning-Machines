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
        self.rob = robobo.SimulationRobobo().connect(address='127.0.0.1', port=19997)
        self.rob.play_simulation()
        self.rob.set_phone_tilt(0.4, 10)
        self.time_since_obj = 0

    def terminate_program(self, signal_number, frame):
        print("Ctrl-C received, terminating program")
        sys.exit(1)

    def play(self, q_table, n_its=20):
        fitness = 0
        for i in range(n_its):
            state, sensors = self.get_state()
            action = random.sample(np.argwhere(q_table[state] == np.amax(q_table[state]))
                                   .flatten().tolist(), 1)[0]

            if action == 0:
                left, right, duration = -15, 15, 300
                reward = 0
            elif action == 1:
                left, right, duration = 15, -15, 300
                reward = 0
            else:  # action == 2:
                left, right, duration = 25, 25, 300
                reward = 5

            self.rob.move(left, right, duration)

            if min(sensors) < 0.2:
                fitness -= 0.5
            else:
                fitness += (reward/n_its)

        return fitness

    def get_state(self):
        sensors = np.log(np.array(self.rob.read_irs())) / 10
        sensors = np.where(sensors == -np.inf, 0, sensors)  # remove the infinite
        sensors = (sensors - -0.65) / 0.65

        n_sensors = []
        for sensor in sensors:
            if sensor < 0.7:
                r = 1
            else:
                r = 0
            n_sensors.append(r)

        r = max(n_sensors[3:6])
        l = max(n_sensors[6:8])
        o = self.detect_object()
        t = 1 if self.time_since_obj < 10 else 0

        return int(str(t)+str(o)+str(l)+str(r), 2), sensors

    def detect_object(self):
        input = self.rob.get_image_front()
        input = input[:, 40:88, :]
        mask = cv2.inRange(input, (0, 100, 0), (90, 255, 90))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if contours:
            self.time_since_obj = 0
            return 1
        else:
            self.time_since_obj += 1
            return 0


ENV = Environment()


class Individual:
    dom_u = 10
    dom_l = 0

    def __init__(self):
        self.age = 0
        self.q_table = list()
        self.fitness = None
        self.time = None
        self.children = 0
        self.parents = None
        self.n_states = 8  # 2bits sensor + 1bit cam detected
        self.n_actions = 3

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

    def evaluate(self):
        self.fitness = ENV.play(q_table=self.q_table)

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
        pickle.dump(self, open('pop2.txt', 'wb'))

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
            individual.evaluate()
            individual.birthday()
            self.individuals.append(individual)

        self.best_individual = self.get_best()
        self.update_stats()
        self.display_population()
        self.mean_fit_history.append(self.mean_fit)
        self.max_fit_history.append(self.max_fit)
        self.worst_fit_history.append(self.worst_fit)
        self.var_history.append(self.var)

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

    def trim(self, type_='fit', elitist=False):
        size = self.size
        new_individuals = list()
        if elitist:
            best = self.get_best()
            best.evaluate()
            size -= 1
            new_individuals.append(best)

        if type_ == 'random':
            new_individuals = random.sample(self.individuals, size)
        elif type_ == 'fit':
            pop_fitness = [(individual.fitness + self.worst_fit) for individual in self.individuals]
            probabilities = [(fit / sum(pop_fitness)) for fit in pop_fitness]
            new_individuals = list(np.random.choice(self.individuals, size=size, replace=False, p=probabilities))
        elif type_ == 'rank':
            pop_fitness = [(individual.fitness + self.worst_fit) for individual in self.individuals]
            ranks = [(sorted(pop_fitness).index(ind) + 1) for ind in pop_fitness]
            probabilities = [rank / sum(ranks) for rank in ranks]
            new_individuals = list(np.random.choice(self.individuals, size=size, replace=False, p=probabilities))

        self.individuals = new_individuals

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
            child.evaluate()
            child.evaluate()
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
        plt.ylim((-5, 5.1))
        plt.ylabel('Fitness')
        plt.xlabel('Generation')
        plt.legend()
        plt.show()


def main(size=5, generations=5, children_per_gen=5, population=None):
    if population is None:
        population = Population(size)
        population.initialize()
    elif population.size < size:
        for j in range(int(size - population.size/2)):
            population.sex(selection_type='rank')
        population.size = size
        population.trim()
    elif population.size > size:
        population.size = size
        population.trim()

    for i in range(generations):
        print('Generation:', population.generation)

        for j in range(children_per_gen):
            population.sex(selection_type='rank')

        population.trim(type_='rank', elitist=False)
        population.display_population()
        population.mutation_rate_change()
        population.next_generation()

    population.plot_generations()



#main(size=10, generations=10, children_per_gen=5)
ind = Individual()
q_table =  (np.array([
                np.array([0, 0, 1]),    # nothing seen in the last few its
                np.array([1, 0, 0]),
                np.array([0, 1, 0]),
                np.array([1, 0, 0]),
                  np.array([0, 0, 1]),  # target detected
                  np.array([0, 0, 1]),
                  np.array([0, 0, 1]),
                  np.array([0, 0, 1]),
                np.array([1, 0, 0]),  # seen something in the last few its
                np.array([1, 0, 0]),
                np.array([0, 1, 0]),
                np.array([1, 0, 0]),
                  np.array([0, 0, 1]),  # target detected
                  np.array([0, 0, 1]),
                  np.array([0, 0, 1]),
                  np.array([0, 0, 1])]))
ind.set_q_table(q_table)
ENV.play(ind.q_table, n_its=2000)

