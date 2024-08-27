import numpy as np
import energy_simulator as es
import math
import random
import latency_simulator as ls


def Energy_sim_annealing(args, epochs, batch_size, fraction, former_accuracy, accuracy, weights, duration,
                         current_energy, energy_sum):
    temperature = 10
    lamb_da = 0.8
    current_energy = current_energy
    print("Initial epochs: {}, batch_size: {}, fraction: {}".format(epochs, batch_size, fraction))

    while temperature > 1.0:
        for i in range(20):
            # add random perturbation to the parameters
            new_epochs = epochs + np.random.randint(-2, 2)
            new_batch_size = batch_size + np.random.randint(-5, 5)
            new_fraction = fraction + np.random.randint(-3, 3) / 100
            new_fraction = round(new_fraction, 2)
            if new_epochs <= 0:
                new_epochs = 1
            elif new_epochs >= 50:
                new_epochs = 50
            if new_batch_size <= 5:
                new_batch_size = 5
            elif new_batch_size >= 500:
                new_batch_size = 500
            if new_fraction <= 0.05:
                new_fraction = 0.05
            if new_fraction >= 0.99:
                new_fraction = 0.99

            new_energy, _ = es.get_local_energy(args, new_epochs, new_batch_size, new_fraction, weights, args.dataset,
                                                duration)
            if new_energy < current_energy:
                epochs = new_epochs
                batch_size = new_batch_size
                fraction = new_fraction
            else:
                p = math.exp(-((new_energy - current_energy) + (former_accuracy - accuracy) - new_energy) / (temperature + sum(energy_sum)))
                if np.random.rand() < p:
                    current_energy = new_energy
                    epochs = new_epochs
                    batch_size = new_batch_size
                    fraction = new_fraction
            print("Updated epochs: {}, batch_size: {}, fraction: {}".format(epochs, batch_size, fraction))
        temperature = lamb_da * temperature
    print("Final epochs: {}, batch_size: {}, fraction: {}".format(epochs, batch_size, fraction))

    return epochs, batch_size, fraction


class Energy_genetic_algorithm:
    def __init__(self, generations=100, epoch_min=1, epoch_max=50, batch_size_min=5, batch_size_max=500,
                 fraction_min=0.05, fraction_max=0.99):
        self.duration = 0.0
        self.generations = generations
        self.group_size = 100
        self.epoch_range = epoch_max - epoch_min
        self.batch_size_range = batch_size_max - batch_size_min
        self.fraction_range = int(fraction_max - fraction_min) * 100
        self.epoch_min = epoch_min
        self.batch_size_min = batch_size_min
        self.fraction_min = fraction_min
        self.crossover_rate = 0.8
        self.mutation_rate = 0.01
        self.para_bits = {
            'epoch': self._bit_length(self.epoch_range),
            'batch_size': self._bit_length(self.batch_size_range),
            'fraction': self._bit_length(self.fraction_range)
        }
        self.dna_size = sum(self.para_bits.values())
        self.DNAs = np.random.randint(2, size=(self.group_size, self.dna_size)).tolist()
        self.deleteDNA = []
        self.deleteFitness = []
        self.accCount = 0

    def _bit_length(self, range_value):
        return (range_value+1).bit_length()

    def translate_DNA(self, epoch_dna, batch_size_dna, fraction_dna):
        epoch = 0
        batch_size = 0
        fraction = 0.0
        # DNAs are stored in Small-Endian
        for i in range(len(epoch_dna)):
            epoch += epoch_dna[i] * 2 ** i
        epoch = epoch / float(2 ** len(epoch_dna) - 1)
        epoch = epoch * self.epoch_range + self.epoch_min
        epoch = round(epoch)
        for i in range(len(batch_size_dna)):
            batch_size += batch_size_dna[i] * 2 ** i
        batch_size = batch_size / float(2 ** len(batch_size_dna) - 1)
        batch_size = batch_size * self.batch_size_range + self.batch_size_min
        batch_size = round(batch_size)
        for i in range(len(fraction_dna)):
            fraction += fraction_dna[i] * 2 ** i
        fraction = fraction / float(2 ** len(fraction_dna) - 1)
        fraction = fraction * self.fraction_range
        fraction = fraction / 100 + self.fraction_min
        fraction = round(fraction, 2)

        return epoch, batch_size, fraction

    def calculate_fitness(self, args, weights, current_energy, energy_sum):
        fitness = []
        delta_energy = []
        total_energy = 0
        idx_epoch = self.para_bits['epoch']
        idx_batch_size = idx_epoch + self.para_bits['batch_size']
        idx_fraction = idx_batch_size + self.para_bits['fraction']
        for DNA in self.DNAs:
            epoch_dna = DNA[:idx_epoch]
            batch_size_dna = DNA[idx_epoch:idx_batch_size]
            fraction_dna = DNA[idx_batch_size:idx_fraction]
            epoch, batch_size, fraction = self.translate_DNA(epoch_dna, batch_size_dna, fraction_dna)
            sim_energy, _ = es.get_local_energy(args, epoch, batch_size, fraction, weights, args.dataset, self.duration)
            delta_energy.append(sim_energy - current_energy)
            total_energy += sim_energy
        for i in range(len(delta_energy)):
            value = math.exp(1 - (delta_energy[i] / total_energy))
            fitness.append(value)

        return fitness

    def select_elite(self, fitness, num_elites):
        elite_dna = []
        for i in range(num_elites):
            index = fitness.index(max(fitness))
            elite_dna.append(self.DNAs[index])
            del fitness[index]
            del self.DNAs[index]

        return elite_dna

    def select(self, fitness):
        selected = random.choices(self.DNAs, weights=fitness, k=2)
        return selected[0], selected[1]

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            crossover_point = np.random.randint(low=1, high=self.dna_size)
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            return child1, child2
        else:
            return parent1, parent2

    def uniform_two_point_crossover(self, parent1, parent2):
        points = sorted(random.sample(range(self.dna_size), 2))
        if np.random.rand() < self.crossover_rate:
            random_number = random.choice([0, 1, 2])
            if random_number == 0:
                child1 = parent2[:points[0]] + parent1[points[0]:]
                child2 = parent1[:points[0]] + parent2[points[0]:]
            elif random_number == 1:
                child1 = parent1[:points[0]] + parent2[points[0]:points[1]] + parent1[points[1]:]
                child2 = parent2[:points[0]] + parent1[points[0]:points[1]] + parent2[points[1]:]
            else:
                child1 = parent1[:points[1]] + parent2[points[1]:]
                child2 = parent2[:points[1]] + parent1[points[1]:]
            return child1, child2
        else:
            return parent1, parent2

    def mutation(self, dna):
        for i in range(len(dna)):
            if np.random.rand() < self.mutation_rate:
                dna[i] = 1 - dna[i]

        return dna

    def scheduler(self, accuracy, fitness):
        accuracy_thresholds = {
            0.8: (20, 20),
            0.7: (40, 40),
            0.6: (60, 60),
            0.3: (80, 80),
            0.0: (100, 100)
        }
        new_group_size = self.group_size
        for acc, (gens, size) in accuracy_thresholds.items():
            if accuracy >= acc:
                self.generations = gens
                new_group_size = size
                break

        if new_group_size < self.group_size:
            for i in range(self.group_size - new_group_size):
                index = fitness.index(min(fitness))
                self.deleteFitness.append(fitness.pop(index))
                self.deleteDNA.append(self.DNAs.pop(index))
        elif new_group_size > self.group_size:
            for i in range(new_group_size - self.group_size):
                index = self.deleteFitness.index(max(self.deleteFitness))
                self.DNAs.append(self.deleteDNA.pop(index))
                fitness.append(self.deleteFitness.pop(index))
        self.group_size = new_group_size

        return fitness

    def run(self, args, epochs, batch_size, fraction, former_accuracy, accuracy, weights, duration, current_energy,
            energy_sum):
        idx_epoch = self.para_bits['epoch']
        idx_batch_size = idx_epoch + self.para_bits['batch_size']
        idx_fraction = idx_batch_size + self.para_bits['fraction']
        self.duration = duration
        current_energy = current_energy
        print("Initial epochs: {}, batch_size: {}, fraction: {}".format(epochs, batch_size, fraction))
        fitness = self.calculate_fitness(args, weights, current_energy, energy_sum)
        fitness = self.scheduler(accuracy, fitness)

        epoch_new, batch_size_new, fraction_new = 0, 0, 0.0
        for generation in range(self.generations):
            new_DNAs = []
            elite_dnas = self.select_elite(fitness, 2)
            new_DNAs.extend(elite_dnas)
            # crossover and mutation
            for _ in range(0, len(self.DNAs), 2):
                parent1, parent2 = self.select(fitness)
                child1, child2 = self.uniform_two_point_crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                new_DNAs.append(child1)
                new_DNAs.append(child2)

            # new generation
            self.DNAs = new_DNAs
            # calculate a new fitness
            fitness = self.calculate_fitness(args, weights, current_energy, energy_sum)
            # choose the best individual of the new generation
            best_solution = max(fitness)
            best_idx = fitness.index(best_solution)
            epoch_new, batch_size_new, fraction_new = self.translate_DNA(self.DNAs[best_idx][:idx_epoch],
                                                                         self.DNAs[best_idx][idx_epoch:idx_batch_size],
                                                                         self.DNAs[best_idx][
                                                                         idx_batch_size:idx_fraction])
            new_energy, _ = es.get_local_energy(args, epochs, batch_size, fraction, weights, args.dataset, self.duration)
            if new_energy < current_energy:
                final_epoch = epoch_new
                final_batch_size = batch_size_new
                final_fraction = fraction_new
            else:
                p = math.exp(-((new_energy - current_energy) + (former_accuracy - accuracy) - new_energy) / (
                        sum(energy_sum)))
                if np.random.rand() < p:
                    final_epoch = epoch_new
                    final_batch_size = batch_size_new
                    final_fraction = fraction_new
                else:
                    final_epoch = epochs
                    final_batch_size = batch_size
                    final_fraction = fraction
            print("Generation: {} \t Epoch: {} \t Batch size: {} \t Fraction: {}".format(generation, final_epoch,
                                                                                         final_batch_size, final_fraction))
        if accuracy < former_accuracy:
            self.accCount += 1
        if self.accCount > 2:
            final_epoch = final_epoch + 2
            final_fraction = final_fraction + 0.05
            self.accCount = 0

        return final_epoch, final_batch_size, final_fraction


def Latency_sim_annealing(args, epochs, batch_size, fraction, former_accuracy, accuracy, weights, duration,
                          current_latency, latency_sum):
    temperature = 10
    lamb_da = 0.9
    current_latency = current_latency
    print("Initial epochs: {}, batch_size: {}, fraction: {}".format(epochs, batch_size, fraction))
    while temperature > 1.0:
        for i in range(20):
            # add random perturbation to the parameters
            new_epochs = epochs + np.random.randint(-2, 2)
            new_batch_size = batch_size + np.random.randint(-5, 5)
            new_fraction = fraction + np.random.randint(-3, 3) / 100
            new_fraction = round(new_fraction, 2)
            if new_epochs <= 0:
                new_epochs = 1
            elif new_epochs >= 50:
                new_epochs = 50
            if new_batch_size <= 5:
                new_batch_size = 5
            elif new_batch_size >= 500:
                new_batch_size = 500
            if new_fraction <= 0.05:
                new_fraction = 0.05
            if new_fraction >= 0.99:
                new_fraction = 0.99

            new_latency, _ = ls.get_local_latency(args, new_epochs, new_batch_size, new_fraction, weights, args.dataset)
            if new_latency < current_latency:
                epochs = new_epochs
                batch_size = new_batch_size
                fraction = new_fraction
            else:
                p = math.exp(-((new_latency - current_latency) + (former_accuracy - accuracy) - new_latency) / (
                        temperature + sum(latency_sum)))
                if np.random.rand() < p:
                    current_latency = new_latency
                    epochs = new_epochs
                    batch_size = new_batch_size
                    fraction = new_fraction
            print("Updated epochs: {}, batch_size: {}, fraction: {}".format(epochs, batch_size, fraction))
        temperature = lamb_da * temperature
    print("Final epochs: {}, batch_size: {}, fraction: {}".format(epochs, batch_size, fraction))

    return epochs, batch_size, fraction


class Latency_genetic_algorithm:
    def __init__(self, generations=100, epoch_min=1, epoch_max=50, batch_size_min=5, batch_size_max=500,
                 fraction_min=0.05, fraction_max=0.99):
        self.generations = generations
        self.group_size = 100
        self.epoch_range = epoch_max - epoch_min
        self.batch_size_range = batch_size_max - batch_size_min
        self.fraction_range = int(fraction_max - fraction_min) * 100
        self.epoch_min = epoch_min
        self.batch_size_min = batch_size_min
        self.fraction_min = fraction_min
        self.crossover_rate = 0.8
        self.mutation_rate = 0.01
        self.para_bits = {
            'epoch': self._bit_length(self.epoch_range),
            'batch_size': self._bit_length(self.batch_size_range),
            'fraction': self._bit_length(self.fraction_range)
        }
        self.dna_size = sum(self.para_bits.values())
        self.DNAs = np.random.randint(2, size=(self.group_size, self.dna_size)).tolist()
        self.deleteDNA = []
        self.deleteFitness = []
        self.accCount = 0

    def _bit_length(self, range_value):
        return (range_value+1).bit_length()

    def translate_DNA(self, epoch_dna, batch_size_dna, fraction_dna):
        epoch = 0
        batch_size = 0
        fraction = 0.0
        for i in range(len(epoch_dna)):
            epoch += epoch_dna[i] * 2 ** i
        epoch = epoch / float(2 ** len(epoch_dna) - 1)
        epoch = epoch * self.epoch_range + self.epoch_min
        epoch = round(epoch)
        for i in range(len(batch_size_dna)):
            batch_size += batch_size_dna[i] * 2 ** i
        batch_size = batch_size / float(2 ** len(batch_size_dna) - 1)
        batch_size = batch_size * self.batch_size_range + self.batch_size_min
        batch_size = round(batch_size)
        for i in range(len(fraction_dna)):
            fraction += fraction_dna[i] * 2 ** i
        fraction = fraction / float(2 ** len(fraction_dna) - 1)
        fraction = fraction * self.fraction_range
        fraction = fraction / 100 + self.fraction_min
        fraction = round(fraction, 2)

        return epoch, batch_size, fraction

    def calculate_fitness(self, args, weights, current_latency):
        fitness = []
        delta_latency = []
        total_latency = 0
        idx_epoch = self.para_bits['epoch']
        idx_batch_size = idx_epoch + self.para_bits['batch_size']
        idx_fraction = idx_batch_size + self.para_bits['fraction']
        for DNA in self.DNAs:
            epoch_dna = DNA[:idx_epoch]
            batch_size_dna = DNA[idx_epoch:idx_batch_size]
            fraction_dna = DNA[idx_batch_size:idx_fraction]
            epoch, batch_size, fraction = self.translate_DNA(epoch_dna, batch_size_dna, fraction_dna)
            sim_latency, _ = ls.get_local_latency(args, epoch, batch_size, fraction, weights, args.dataset)
            delta_latency.append(sim_latency - current_latency)
            total_latency += sim_latency
        for i in range(len(delta_latency)):
            value = math.exp(1 - (delta_latency[i] / total_latency))
            fitness.append(value)

        return fitness

    def select_elite(self, fitness, num_elites):
        elite_dna = []
        for i in range(num_elites):
            index = fitness.index(max(fitness))
            elite_dna.append(self.DNAs[index])
            del fitness[index]
            del self.DNAs[index]

        return elite_dna

    def select(self, fitness):
        selected = random.choices(self.DNAs, weights=fitness, k=2)
        return selected[0], selected[1]

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            crossover_point = np.random.randint(low=1, high=self.dna_size)
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            return child1, child2
        else:
            return parent1, parent2

    def uniform_two_point_crossover(self, parent1, parent2):
        points = sorted(random.sample(range(self.dna_size), 2))
        if np.random.rand() < self.crossover_rate:
            random_number = random.choice([0, 1, 2])
            if random_number == 0:
                child1 = parent2[:points[0]] + parent1[points[0]:]
                child2 = parent1[:points[0]] + parent2[points[0]:]
            elif random_number == 1:
                child1 = parent1[:points[0]] + parent2[points[0]:points[1]] + parent1[points[1]:]
                child2 = parent2[:points[0]] + parent1[points[0]:points[1]] + parent2[points[1]:]
            else:
                child1 = parent1[:points[1]] + parent2[points[1]:]
                child2 = parent2[:points[1]] + parent1[points[1]:]
            return child1, child2
        else:
            return parent1, parent2

    def mutation(self, dna):
        for i in range(len(dna)):
            if np.random.rand() < self.mutation_rate:
                dna[i] = 1 - dna[i]

        return dna

    def scheduler(self, accuracy, fitness):
        accuracy_thresholds = {
            0.8: (20, 20),
            0.7: (40, 40),
            0.6: (60, 60),
            0.3: (80, 80),
            0.0: (100, 100)
        }
        new_group_size = self.group_size
        for acc, (gens, size) in accuracy_thresholds.items():
            if accuracy >= acc:
                self.generations = gens
                new_group_size = size
                break

        if new_group_size < self.group_size:
            for i in range(self.group_size - new_group_size):
                index = fitness.index(min(fitness))
                self.deleteFitness.append(fitness.pop(index))
                self.deleteDNA.append(self.DNAs.pop(index))
        elif new_group_size > self.group_size:
            for i in range(new_group_size - self.group_size):
                index = self.deleteFitness.index(max(self.deleteFitness))
                self.DNAs.append(self.deleteDNA.pop(index))
                fitness.append(self.deleteFitness.pop(index))
        self.group_size = new_group_size

        return fitness

    def run(self, args, epochs, batch_size, fraction, former_accuracy, accuracy, weights, current_latency, latency_sum):
        idx_epoch = self.para_bits['epoch']
        idx_batch_size = idx_epoch + self.para_bits['batch_size']
        idx_fraction = idx_batch_size + self.para_bits['fraction']
        current_latency = current_latency
        print("Initial epochs: {}, batch_size: {}, fraction: {}".format(epochs, batch_size, fraction))
        fitness = self.calculate_fitness(args, weights, current_latency)
        fitness = self.scheduler(accuracy, fitness)

        epoch_new, batch_size_new, fraction_new = 0, 0, 0.0
        for generation in range(self.generations):
            new_DNAs = []
            elite_dnas = self.select_elite(fitness, 2)
            new_DNAs.extend(elite_dnas)
            # crossover and mutation
            for _ in range(0, len(self.DNAs), 2):
                parent1, parent2 = self.select(fitness)
                child1, child2 = self.uniform_two_point_crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                new_DNAs.append(child1)
                new_DNAs.append(child2)

            # new generation
            self.DNAs = new_DNAs
            # calculate a new fitness
            fitness = self.calculate_fitness(args, weights, current_latency)
            # choose the best individual of the new generation
            best_solution = max(fitness)
            best_idx = fitness.index(best_solution)
            epoch_new, batch_size_new, fraction_new = self.translate_DNA(self.DNAs[best_idx][:idx_epoch],
                                                                         self.DNAs[best_idx][idx_epoch:idx_batch_size],
                                                                         self.DNAs[best_idx][
                                                                         idx_batch_size:idx_fraction])
            new_latency, _ = ls.get_local_latency(args, epoch_new, batch_size_new, fraction_new, weights, args.dataset)
            if new_latency < current_latency:
                final_epoch = epoch_new
                final_batch_size = batch_size_new
                final_fraction = fraction_new
            else:
                p = math.exp(-((new_latency - current_latency) + (former_accuracy - accuracy) - new_latency) / (
                        sum(latency_sum)))
                if np.random.rand() < p:
                    final_epoch = epoch_new
                    final_batch_size = batch_size_new
                    final_fraction = fraction_new
                else:
                    final_epoch = epochs
                    final_batch_size = batch_size
                    final_fraction = fraction
            print("Generation: {} \t Epoch: {} \t Batch size: {} \t Fraction: {}".format(generation, final_epoch,
                                                                                         final_batch_size, final_fraction))
        if accuracy < former_accuracy:
            self.accCount += 1
        if self.accCount > 2:
            final_epoch = final_epoch + 2
            final_fraction = final_fraction + 0.05
            self.accCount = 0

        return final_epoch, final_batch_size, final_fraction
