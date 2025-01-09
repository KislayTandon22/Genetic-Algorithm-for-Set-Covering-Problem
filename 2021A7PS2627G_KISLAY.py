import random
import time
from SetCoveringProblemCreator import *

MAX_TIME = 40

def initialize_population(size, subsets):
    population = []
    for _ in range(size):
        individual = [random.choice([0, 1]) for _ in range(len(subsets))]
        population.append(individual)
    return population

def fitness_function(individual, subsets):
    covered = set()
    count = 0
    for i, bit in enumerate(individual):
        if bit:
            covered.update(subsets[i])
            count += 1
    coverage = len(covered) / 100  # Adjust universe size here if needed
    return (coverage - (count / len(subsets))) * 100

def select_parents(population, fitnesses):
    total_fitness = sum(fitnesses)
    if total_fitness == 0:
        total_fitness = 1  # Avoid division by zero
    probabilities = [f / total_fitness for f in fitnesses]
    parent1, parent2 = random.choices(population, weights=probabilities, k=2)
    return parent1, parent2

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(individual):
    i = random.randint(0, len(individual) - 1)
    individual[i] = 1 - individual[i]

def genetic_algorithm(subsets, population_size=100, generations=50, start_time=None, elitism_rate=0.1):
    population = initialize_population(population_size, subsets)
    best_fitness_over_time = []
    mean_subset_size_over_time = []

    best_solution = None
    best_fitness = float('-inf')

    initial_mutation_rate = 0.3
    final_mutation_rate = 0.01

    for generation in range(generations):
        if time.time() - start_time > MAX_TIME:
            print("Time limit reached.")
            break

        current_mutation_rate = initial_mutation_rate - (initial_mutation_rate - final_mutation_rate) * (generation / generations)
        
        fitnesses = [fitness_function(ind, subsets) for ind in population]
        
        for i, fitness in enumerate(fitnesses):
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = population[i]
        
        best_fitness_over_time.append(best_fitness)
        mean_subset_size_over_time.append(sum(best_solution))
        
        # Introduce elitism
        num_elites = int(elitism_rate * population_size)
        sorted_population = [x for _, x in sorted(zip(fitnesses, population), reverse=True)]
        elites = sorted_population[:num_elites]

        population2 = elites.copy()
        while len(population2) < population_size:
            parent1, parent2 = select_parents(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            if random.random() < current_mutation_rate:
                mutate(child1)
            if random.random() < current_mutation_rate:
                mutate(child2)
            population2.append(child1)
            population2.append(child2)
        
        population = population2[:population_size]  # Ensure population size remains constant

    return best_solution, best_fitness_over_time, mean_subset_size_over_time


def main():
    scp = SetCoveringProblemCreator()

    # Load the SCP problem from the JSON file
    listOfSubsets = scp.ReadSetsFromJson("scp_test.json")

    # Start timing
    start_time = time.time()

    # Run the genetic algorithm on the problem loaded from JSON
    best_solution, best_fitness_over_time, mean_subset_size_over_time = genetic_algorithm(listOfSubsets, population_size=50, generations=1000000, start_time=start_time)
    
    print(f"Number of sets: {len(best_solution)}")
    print("Solution:", end=" ")
    for i, bit in enumerate(best_solution):
        print(f"{i}:{bit}", end=", ")
    print()
    print(f"Fitness value of best state: {fitness_function(best_solution, listOfSubsets)}")
    print(f"Minimum number of subsets that can cover the universe set: {sum(best_solution)}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.3f} seconds")


if __name__ == '__main__':
    main()
