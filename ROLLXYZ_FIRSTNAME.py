import random
import time
from SetCoveringProblemCreator import *
import matplotlib.pyplot as plt
import numpy as np
import os

MAX_TIME = 45

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

def genetic_algorithm(subsets, population_size=100, generations=50, start_time=None):
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
        
        population2 = []
        for _ in range(population_size ):
            parent1, parent2 = select_parents(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            if random.random() < current_mutation_rate:
                mutate(child1)
            if random.random() < current_mutation_rate:
                mutate(child2)
            population2.append(child1)
            population2.append(child2)
        
        population = population2

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

        mean_best_fitness_over_time = np.mean(fitness_over_time_all_runs, axis=0)
        std_best_fitness = np.std([run[-1] for run in fitness_over_time_all_runs])

        results[size] = {
            'mean_best_fitness': np.mean([run[-1] for run in fitness_over_time_all_runs]),
            'std_best_fitness': std_best_fitness,
            'mean_fitness_over_time': mean_best_fitness_over_time,
            'mean_subset_size_over_time': np.mean(mean_subset_size_over_time_all_runs, axis=0)
        }

    return results

    



def plot_experiment_results(results, generations=50, output_dir='plots'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sizes = sorted(results.keys())
    
    # Plot mean and standard deviation of best fitness value after specified generations
    means = [results[size]['mean_best_fitness'] for size in sizes]
    std_devs = [results[size]['std_best_fitness'] for size in sizes]

    plt.figure(figsize=(10, 6))
    plt.errorbar(sizes, means, yerr=std_devs, fmt='-o', capsize=5, capthick=2, label='Mean Best Fitness ± Std. Dev.')

    for i, size in enumerate(sizes):
        plt.annotate(f'{means[i]:.2f} ± {std_devs[i]:.2f}', 
                     (size, means[i]), 
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center')

    plt.title(f'Mean and Std. Dev. of Best Fitness after {generations} Generations')
    plt.xlabel('Number of Subsets')
    plt.ylabel('Fitness Value')
    plt.xticks(sizes)
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'mean_std_best_fitness.png'))
    
    # Plot how the mean best fitness value changes over the generations for different subset sizes
    plt.figure(figsize=(12, 8))
    for size in sizes:
        mean_fitness = results[size]['mean_fitness_over_time']
        plt.plot(mean_fitness, label=f'|S| = {size}')
        
        # Annotate start and end points
        plt.annotate(f'{mean_fitness[0]:.2f}', (0, mean_fitness[0]), textcoords="offset points", xytext=(0,10), ha='center')
        plt.annotate(f'{mean_fitness[-1]:.2f}', (generations - 1, mean_fitness[-1]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.title(f'Mean Best Fitness Value over {generations} Generations')
    plt.xlabel('Generations')
    plt.ylabel('Mean Fitness Value')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'mean_fitness_over_time.png'))
    
    # Plot how the mean subset size changes over the generations for different subset sizes
    plt.figure(figsize=(12, 8))
    for size in sizes:
        mean_subset_size = results[size]['mean_subset_size_over_time']
        plt.plot(mean_subset_size, label=f'|S| = {size}')
        
        # Annotate start and end points
        plt.annotate(f'{mean_subset_size[0]:.2f}', (0, mean_subset_size[0]), textcoords="offset points", xytext=(0,10), ha='center')
        plt.annotate(f'{mean_subset_size[-1]:.2f}', (generations - 1, mean_subset_size[-1]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.title(f'Mean Subset Size for Highest Fitness over {generations} Generations')
    plt.xlabel('Generations')
    plt.ylabel('Mean Subset Size')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'mean_subset_size_over_time.png'))
    



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
        
        # Plotting results for the individual run
        plt.figure(figsize=(10, 6))
        plt.plot(best_fitness_over_time, label='Best Fitness Over Time')
        plt.title(f'Best Fitness Over Time for Subset Size {size}')
        plt.xlabel('Generations')
        plt.ylabel('Best Fitness Value')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'plots/best_fitness_over_time_{size}.png')
        
        
        plt.figure(figsize=(10, 6))
        plt.plot(mean_subset_size_over_time, label='Mean Subset Size Over Time')
        plt.title(f'Mean Subset Size Over Time for Subset Size {size}')
        plt.xlabel('Generations')
        plt.ylabel('Mean Subset Size')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'plots/mean_subset_size_over_time_{size}.png')
        

    elif option == "2":
        generations = 250
        results = run_experiment(start_time=start_time, generations=generations)
        plot_experiment_results(results, generations=generations)
    
    else:
        print("Invalid option. Please enter 1 or 2.")
        sys.exit(1)

    end_time = time.time()  # End timing after the operation is completed
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.3f} seconds")

if __name__ == '__main__':
    main()
