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
    coverage = len(covered) / 100  # Assuming universe size is always 100
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
    child = parent1[:point] + parent2[point:]
    return child

def mutate(individual):
    i = random.randint(0, len(individual) - 1)
    individual[i] = 1 - individual[i]

def genetic_algorithm(subsets, population_size=100, generations=1000, 
                      crossover_rate=0.8, mutation_rate=0.01, start_time=None):
    
    population = initialize_population(population_size, subsets)
    best_fitness_over_time = []
    min_subsets_over_time = []

    for generation in range(generations):
        if time.time() - start_time > MAX_TIME:
            print("Time limit reached.")
            break

        fitnesses = [fitness_function(ind, subsets) for ind in population]
        best_index = fitnesses.index(max(fitnesses))
        best_individual = population[best_index]
        best_fitness_over_time.append(max(fitnesses))
        min_subsets_over_time.append(sum(best_individual))

        new_population = [best_individual]  # Elitism

        while len(new_population) < population_size:
            parent1, parent2 = select_parents(population, fitnesses)
            
            if random.random() < crossover_rate:
                child = crossover(parent1, parent2)
            else:
                child = parent1.copy()
            mutate(child)
            new_population.append(child)

        population = new_population

    if not best_fitness_over_time:
        best_fitness_over_time = [0]
        min_subsets_over_time = [0]

    best_solution = max(population, key=lambda ind: fitness_function(ind, subsets))
    return best_solution, best_fitness_over_time, min_subsets_over_time

def run_experiment(start_time):
    population_sizes = [50]
    generations_list = [50]
    mutation_rates = [0.5]
    num_experiments = 10
    scp_sizes = [50, 150, 250, 350]
    results = []

    for size in scp_sizes:
        for pop_size in population_sizes:
            for gen_count in generations_list:
                for mutation_rate in mutation_rates:
                    mean_fitnesses = []
                    all_min_subsets_over_time = []

                    for _ in range(num_experiments):
                        scp = SetCoveringProblemCreator()
                        subsets = scp.Create(usize=100, totalSets=size)
                        best_solution, fitness_over_time, min_subsets_over_time = genetic_algorithm(
                            subsets, population_size=pop_size, generations=gen_count, mutation_rate=mutation_rate, start_time=start_time)
                        mean_fitnesses.append(fitness_over_time[-1])
                        all_min_subsets_over_time.append(min_subsets_over_time)

                    avg_fitness = np.mean(mean_fitnesses)
                    avg_min_subsets = np.mean(all_min_subsets_over_time, axis=0)
                    results.append({
                        'Subset Size': size,
                        'Population Size': pop_size,
                        'Generations': gen_count,
                        'Mutation Rate': mutation_rate,
                        'Avg Final Fitness': avg_fitness,
                        'Best Fitness Over Generations': fitness_over_time,
                        'Min Subsets Over Time': avg_min_subsets
                    })

    return results

def plot_experiment_results(results, output_dir='plots'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    valid_results = [result for result in results if result and 'Subset Size' in result]

    if not valid_results:
        print("No valid results found to plot.")
        return

    subset_sizes = sorted(set(result['Subset Size'] for result in valid_results))

    # Arrays to store mean and standard deviation values
    mean_fitness_values = []
    std_fitness_values = []

    for size in subset_sizes:
        filtered_results = [result for result in valid_results if result['Subset Size'] == size]
        fitness_values_per_run = [result['Best Fitness Over Generations'][-1] for result in filtered_results]

        # Calculate mean and standard deviation for the last generation
        mean_fitness = np.mean(fitness_values_per_run)
        std_fitness = np.std(fitness_values_per_run)

        mean_fitness_values.append(mean_fitness)
        std_fitness_values.append(std_fitness)

    # Plot Mean Fitness Values with Standard Deviation
    plt.figure(figsize=(10, 6))
    plt.errorbar(subset_sizes, mean_fitness_values, yerr=std_fitness_values, fmt='-o', capsize=5, ecolor='red')
    plt.title('Mean Fitness Values with Standard Deviation for Different Subset Sizes')
    plt.xlabel('Subset Size')
    plt.ylabel('Fitness Value')
    plt.grid(True)
    plt.savefig(f"{output_dir}/mean_fitness_with_std_dev.png")
    plt.close()
    print(f"Saved graph: {output_dir}/mean_fitness_with_std_dev.png")

    # Plot Mean Best Fitness Over Generations with Standard Deviation
    plt.figure(figsize=(10, 6))
    for size in subset_sizes:
        filtered_results = [result for result in valid_results if result['Subset Size'] == size]
        fitness_values_per_run = np.array([result['Best Fitness Over Generations'] for result in filtered_results])
        mean_fitness_over_generations = np.mean(fitness_values_per_run, axis=0)
        std_fitness_over_generations = np.std(fitness_values_per_run, axis=0)

        plt.plot(mean_fitness_over_generations, label=f'Subset Size {size}')
        plt.fill_between(range(len(mean_fitness_over_generations)),
                         mean_fitness_over_generations - std_fitness_over_generations,
                         mean_fitness_over_generations + std_fitness_over_generations, alpha=0.2)
    
    plt.title('Mean Best Fitness Over Generations with Standard Deviation')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/mean_fitness_over_generations_with_std_dev.png")
    plt.close()
    print(f"Saved graph: {output_dir}/mean_fitness_over_generations_with_std_dev.png")



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
        
        listOfSubsets = scp.ReadSetsFromJson("scp_test.json")
        best_solution, best_fitness_over_time, min_subsets_over_time = genetic_algorithm(listOfSubsets, start_time=start_time)
        print(f"Number of sets: {len(listOfSubsets)}/{len(subsets)}")
        print("Solution:", end=" ")
        for i, bit in enumerate(best_solution):
            print(f"{i}:{bit}", end=", ")
        print()
        print(f"Fitness value of best state: {fitness_function(best_solution, listOfSubsets)}")
        print(f"Minimum number of subsets that can cover the universe set: {sum(best_solution)}")
    
    elif option == "2":
        results = run_experiment(start_time=start_time)
        plot_experiment_results(results)
    
    else:
        print("Invalid option. Please enter 1 or 2.")
        sys.exit(1)

    end_time = time.time()  # End timing after the operation is completed
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.3f} seconds")

if __name__ == '__main__':
    main()
