import random
import time
from SetCoveringProblemCreator import *
import os

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
    coverage = len(covered) / 100 
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

def run_experiment(start_time, generations=50):
    subset_sizes = [50, 150, 250, 350]
    results = {}

    for size in subset_sizes:
        fitness_over_time_all_runs = []
        mean_subset_size_over_time_all_runs = []

        scp = SetCoveringProblemCreator()

        for _ in range(10):  
            subsets = scp.Create(usize=100, totalSets=size)
            _, best_fitness_over_time, mean_subset_size_over_time = genetic_algorithm(subsets, population_size=50, generations=generations, start_time=start_time)

            fitness_over_time_all_runs.append(best_fitness_over_time)
            mean_subset_size_over_time_all_runs.append(mean_subset_size_over_time)

        # Calculate mean and std using simple Python
        mean_best_fitness_over_time = [sum(run) / len(run) for run in zip(*fitness_over_time_all_runs)]
        mean_fitness = sum([run[-1] for run in fitness_over_time_all_runs]) / len(fitness_over_time_all_runs)
        std_best_fitness = (sum((run[-1] - mean_fitness) ** 2 for run in fitness_over_time_all_runs) / len(fitness_over_time_all_runs)) ** 0.5

        results[size] = {
            'mean_best_fitness': mean_fitness,
            'std_best_fitness': std_best_fitness,
            'mean_fitness_over_time': mean_best_fitness_over_time,
            'mean_subset_size_over_time': [sum(run) / len(run) for run in zip(*mean_subset_size_over_time_all_runs)]
        }

    return results

def main():
    print("Choose an option:")
    print("1. Individual Run")
    print("2. Batch Run")

    option = input("Enter your choice (1 or 2): ").strip()

    start_time = time.time()  # Start timing after user input

    if option == "1":
        size = int(input("Enter the subset size (e.g., 50, 150, 250, 350): "))
        scp = SetCoveringProblemCreator()
        subsets = scp.Create(usize=100, totalSets=size)
        
        best_solution, best_fitness_over_time, mean_subset_size_over_time = genetic_algorithm(subsets, population_size=50, generations=1000000, start_time=start_time)
        
        print(f"Number of sets: {len(best_solution)}")
        print("Solution:", end=" ")
        for i, bit in enumerate(best_solution):
            print(f"{i}:{bit}", end=", ")
        print()
        print(f"Fitness value of best state: {fitness_function(best_solution, subsets)}")
        print(f"Minimum number of subsets that can cover the universe set: {sum(best_solution)}")

    elif option == "2":
        generations = 250
        results = run_experiment(start_time=start_time, generations=generations)
    
    else:
        print("Invalid option. Please enter 1 or 2.")
        sys.exit(1)

    end_time = time.time()  # End timing after the operation is completed
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.3f} seconds")

if __name__ == '__main__':
    main()
