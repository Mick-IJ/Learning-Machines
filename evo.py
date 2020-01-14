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


def sigmoid_activation(x):
    return 1./(1.+np.exp(-x))


class Controller:
    def __init__(self):
        self.n_hidden = [10]
        self.n_output = 2

    def control(self, inputs, controller):
        if self.n_hidden[0] > 0:
            # Preparing the weights and biases from the controller of layer 1

            # Biases for the n hidden neurons
            bias1 = controller[:self.n_hidden[0]].reshape(1, self.n_hidden[0])
            # Weights for the connections from the inputs to the hidden nodes
            weights1_slice = len(inputs) * self.n_hidden[0] + self.n_hidden[0]
            weights1 = controller[self.n_hidden[0]:weights1_slice].reshape((len(inputs), self.n_hidden[0]))

            # Outputs activation first layer.
            output1 = sigmoid_activation(inputs.dot(weights1) + bias1)

            # Preparing the weights and biases from the controller of layer 2
            bias2 = controller[weights1_slice:weights1_slice + self.n_output].reshape(1, self.n_output)
            weights2 = controller[weights1_slice + self.n_output:].reshape((self.n_hidden[0], self.n_output))

            # Outputting activated second layer. Each entry in the output is an action
            output = sigmoid_activation(output1.dot(weights2) + bias2)[0]
        else:
            bias = controller[:self.n_output].reshape(1, self.n_output)
            weights = controller[self.n_output:].reshape((len(inputs), self.n_output))

            output = sigmoid_activation(inputs.dot(weights) + bias)[0]

        return (output*2) - 1


class Environment:
    def __init__(self, controller):
        self.num_sensors = 8
        self.controller = controller
        signal.signal(signal.SIGINT, self.terminate_program)

    def terminate_program(self, signal_number, frame):
        print("Ctrl-C received, terminating program")
        sys.exit(1)

    def play(self, p_cont):
        for robobo_num in ['', '#0', '#2']:

            self.rob = robobo.SimulationRobobo(number=robobo_num).connect(address='127.0.0.1', port=19997)
            self.rob.play_simulation()

            fitness = 0
            n_its = 20
            for i in range(n_its):
                inputs = np.log(np.asarray(self.rob.read_irs())) / 10  # read and scale
                inputs = np.where(inputs == -np.inf, 0, inputs)  # remove the infinite
                inputs = (inputs - -0.65) / (0 - -0.65)  # scale between 0 and 1

                action = self.controller.control(inputs, p_cont)
                left, right = action[0]*50, action[1]*50

                self.rob.move(left, right, 200)
                s_trans = abs(left) + abs(right)
                s_rot = abs(left - right) / 100
                v_sens = min(inputs)

                fitness += ((s_trans*(1-s_rot)*(v_sens)) / n_its)

            self.rob.disconnect()

        self.rob.stop_world()
        return fitness/3


ENV = Environment(Controller())


class Individual:
    dom_u = 1
    dom_l = -1
    n_hidden = 10
    n_vars = (ENV.num_sensors + 1) * n_hidden + (n_hidden + 1) * 2  # multilayer with 50 neurons

    def __init__(self):
        self.age = 0
        self.weights = list()
        self.fitness = None
        self.time = None
        self.children = 0
        self.parents = None

    def set_weights(self, weights=None):
        if weights is None:
            self.weights = np.random.uniform(self.dom_l, self.dom_u, self.n_vars)
        else:
            self.weights = weights

    def evaluate(self):
        self.fitness = ENV.play(p_cont=self.weights)

    def check_limits(self):
        new_weights = list()
        for weight in self.weights:
            if weight > self.dom_u:
                new_weights.append(self.dom_u)
            elif weight < self.dom_l:
                new_weights.append(self.dom_l)
            else:
                new_weights.append(weight)

        self.weights = np.asarray(new_weights)

    def birthday(self):
        self.age += 1

    def mutate(self, mutation_rate):
        for i in range(0, len(self.weights)):
            # if np.random.random() <= mutation_rate ** 2:
            #     self.weights[i] = np.random.normal(0, 1)
            if np.random.random() <= mutation_rate:
                self.weights[i] = self.weights[i] * np.random.normal(0, 1.27)
            # if np.random.random() <= mutation_rate:
            #     self.weights[i] = self.weights[i] + np.random.normal(0, .1)
        # if np.random.random() <= mutation_rate ** 3:
        #     np.random.shuffle(self.weights)
        self.check_limits()


class Population:
    def __init__(self, size=5):
        self.individuals = list()
        self.size = size
        self.generation = 1
        self.mutation_rate = 0.3
        self.mean_age = None
        self.mean_children = None
        self.mean_fit = None
        self.max_fit = None
        self.worst_fit = None
        self.mean_fit_history = list()
        self.max_fit_history = list()
        self.worst_fit_history = list()

    def append(self, individual):
        self.individuals.append(individual)
        self.update_stats()

    def extend(self, population):
        self.individuals.extend(population)
        self.update_stats()

    def kill(self, individual):
        self.individuals.remove(individual)
        self.update_stats()

    def update_stats(self):
        population_fit = [i.fitness for i in self.individuals]
        self.mean_fit = np.mean(population_fit)
        self.max_fit = np.max(population_fit)
        self.worst_fit = np.min(population_fit)
        self.mean_age = np.mean([i.age for i in self.individuals])
        self.mean_children = np.mean([i.children for i in self.individuals])

    def display_population(self):
        i = 1
        for individual in self.individuals:
            if individual.parents is not None:
                parent1, parent2 = individual.parents
                mean_parents = (parent1.fitness + parent2.fitness) / 2
            else:
                mean_parents = 0
            print(i, ': fitness =', round(individual.fitness, 4), 'age =', individual.age,
                  'children =', individual.children, 'parent_fit =', round(mean_parents, 2))
            i += 1

        print('Mean fitness:', round(self.mean_fit, 4), 'Mean age:', round(self.mean_age, 2),
              'Mean children =', round(self.mean_children, 2), '\n')

    def initialize(self):
        for i in range(self.size):
            individual = Individual()
            individual.set_weights()
            individual.evaluate()
            individual.birthday()
            self.individuals.append(individual)

        self.update_stats()
        self.display_population()

    def select_parents(self, n, type_='random'):
        if type_ == 'random':
            return sample(self.individuals, n)
        elif type_ == 'fit':
            pop_fitness = [(individual.fitness + 7) for individual in self.individuals]
            probabilities = [(fit / sum(pop_fitness)) for fit in pop_fitness]
            return np.random.choice(self.individuals, size=n, replace=False, p=probabilities)
        elif type_ == 'rank':
            pop_fitness = [(individual.fitness + 7) for individual in self.individuals]
            ranks = [sorted(pop_fitness).index(ind) for ind in pop_fitness]
            probabilities = [rank / sum(ranks) for rank in ranks]
            return np.random.choice(self.individuals, size=n, replace=False, p=probabilities)

    def trim(self, type_='fit'):
        if type_ == 'random':
            self.individuals = sample(self.individuals, self.size)
        elif type_ == 'fit':
            pop_fitness = [(individual.fitness + 7) for individual in self.individuals]
            probabilities = [(fit / sum(pop_fitness)) for fit in pop_fitness]
            self.individuals = list(np.random.choice(self.individuals, size=self.size, replace=False, p=probabilities))
        elif type_ == 'rank':
            pop_fitness = [(individual.fitness + 7) for individual in self.individuals]
            ranks = [(sorted(pop_fitness).index(ind) + 0.1) for ind in pop_fitness]
            probabilities = [rank / sum(ranks) for rank in ranks]
            self.individuals = list(np.random.choice(self.individuals, size=self.size, replace=False, p=probabilities))

        self.update_stats()

    def sex(self, type_='mean', selection_type='fit'):

        parent1, parent2 = self.select_parents(2, type_=selection_type)

        cross_prop = np.random.random()

        if type_ == 'mean':
            children = [np.array(parent1.weights) * cross_prop + np.array(parent2.weights) * (1 - cross_prop)]

        elif type_ == 'recombine':
            split_loc = int(len(parent1.weights) * cross_prop)
            child1 = np.append(parent1.weights[:split_loc], parent2.weights[split_loc:])
            child2 = np.append(parent2.weights[:split_loc], parent1.weights[split_loc:])
            children = [child1, child2]

        else:  # type == 'uniform'
            child = list(np.zeros(len(parent1.weights)))
            for i in range(len(child)):
                if np.random.random() <= .5:
                    child[i] = parent1.weights[i]
                else:
                    child[i] = parent2.weights[i]
            children = [child]

        for child_weights in children:
            child = Individual()
            child.set_weights(child_weights)
            child.mutate(mutation_rate=self.mutation_rate)
            child.evaluate()
            child.parents = (parent1, parent2)
            self.individuals.append(child)

            parent1.children += 1
            parent2.children += 1

    def mutation_rate_change(self, type_='exp'):
        if type_ == 'linear':
            self.mutation_rate -= 0.01
        elif type_ == 'exp':
            self.mutation_rate *= 0.99
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
        plt.ylim((0, 50))
        plt.legend()
        plt.title("First run score")
        plt.show()


def main(size=5, generations=5, children_per_gen=5):
    population = Population(size)
    population.initialize()

    for i in range(generations):
        print('Generation:', population.generation)

        for j in range(children_per_gen):
            population.sex(type_='recombine', selection_type='rank')
    
        population.trim(type_='rank')
        population.display_population()

        population.mutation_rate_change()
        population.next_generation()

    population.plot_generations()


main(size=5, generations=5)
