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
        self.rob = robobo.SimulationRobobo(number='#0').connect(address='127.0.0.1', port=19997)
        self.rob.play_simulation()

    def terminate_program(self, signal_number, frame):
        print("Ctrl-C received, terminating program")
        sys.exit(1)

    def play(self, q_table):
        fitness = 0
        n_its = 50
        for i in range(n_its):
            state, sensors = self.get_state()
            action = random.sample(np.argwhere(q_table[state] == np.amax(q_table[state]))
                                   .flatten().tolist(), 1)[0]

            if action == 0:
                left, right, duration = 25, 25, 200
                reward = 10
            elif action == 1:
                left, right, duration = 10, 25, 300
                reward = 5
            elif action == 2:
                left, right, duration = 25, 10, 300
                reward = 5
            elif action == 3:
                left, right, duration = -10, -25, 300
                reward = 1
            else:  # action == 3:
                left, right, duration = -25, -10, 300
                reward = 1

            self.rob.move(left, right, duration)
            v_sens = min(sensors)

            if min(sensors) < 0.1:
                fitness -= 1
            else:
                fitness += (reward/ n_its)

        return fitness

    def get_state(self):
        sensors = np.log(np.array(self.rob.read_irs())) / 10
        sensors = np.where(sensors == -np.inf, 0, sensors)  # remove the infinite
        sensors = (sensors - -0.65) / 0.65

        n_sensors = []
        for sensor in sensors:
            if sensor < 0.6:
                r = 1
            else:
                r = 0
            n_sensors.append(r)

        bl = max(n_sensors[:2])
        br = max(n_sensors[1:3])
        fl = max(n_sensors[4:7])
        fr = max(n_sensors[6:8])

        return int(str(bl)+str(br)+str(fl)+str(fr), 2), sensors


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
        self.n_states = 32
        self.n_actions = 5

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
        self.mutation_rate = 0.2
        self.mean_age = None
        self.mean_children = None
        self.mean_fit = None
        self.max_fit = None
        self.worst_fit = None
        self.mean_fit_history = list()
        self.max_fit_history = list()
        self.worst_fit_history = list()
        self.best_individual = None

    def append(self, individual):
        self.individuals.append(individual)
        self.update_stats()

    def extend(self, population):
        self.individuals.extend(population)
        self.update_stats()

    def kill(self, individual):
        self.individuals.remove(individual)
        self.update_stats()

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
        self.mean_fit = np.mean(population_fit)
        self.max_fit = np.max(population_fit)
        self.worst_fit = np.min(population_fit)
        self.mean_age = np.mean([i.age for i in self.individuals])
        self.mean_children = np.mean([i.children for i in self.individuals])
        self.best_individual = self.get_best()
        pickle.dump(self, open('pop.txt', 'wb'))

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

    def select_parents(self, n, type_='random'):
        if type_ == 'random':
            return random.sample(self.individuals, n)
        elif type_ == 'fit':
            pop_fitness = [(individual.fitness + 50) for individual in self.individuals]
            probabilities = [(fit / sum(pop_fitness)) for fit in pop_fitness]
            return np.random.choice(self.individuals, size=n, replace=False, p=probabilities)
        elif type_ == 'rank':
            pop_fitness = [(individual.fitness + 50) for individual in self.individuals]
            ranks = [sorted(pop_fitness).index(ind) for ind in pop_fitness]
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
            pop_fitness = [(individual.fitness + 50) for individual in self.individuals]
            probabilities = [(fit / sum(pop_fitness)) for fit in pop_fitness]
            new_individuals = list(np.random.choice(self.individuals, size=size, replace=False, p=probabilities))
        elif type_ == 'rank':
            pop_fitness = [(individual.fitness + 50) for individual in self.individuals]
            ranks = [(sorted(pop_fitness).index(ind) + 0.1) for ind in pop_fitness]
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

    def plot_generations(self):
        plt.figure(figsize=(12, 12))
        plt.plot(self.max_fit_history, label="best")
        plt.plot(self.mean_fit_history, label="avg")
        plt.plot(self.worst_fit_history, label='worst')
        plt.ylim((-20, 20))
        plt.legend()
        plt.title("First run score")
        plt.show()


def main(size=5, generations=5, children_per_gen=5):
    population = Population(size)
    population.initialize()

    for i in range(generations):
        print('Generation:', population.generation)

        for j in range(children_per_gen):
            population.sex(selection_type='rank')

        population.trim(type_='fit', elitist=True)
        population.display_population()
        population.mutation_rate_change()
        population.next_generation()

    population.plot_generations()
    pickle.dump(population, open('pop.txt', 'wb'))


main(size=10, generations=10, children_per_gen=5)
