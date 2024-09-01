import random
import time
from SetCoveringProblemCreator import *
import matplotlib.pyplot as plt
import numpy as np

def initialize_population(size, subsets):
    population = []
    for _ in range(size):
        individual = [random.choice([0, 1]) for _ in range(len(subsets))]
        population.append(individual)
    return population

def fitness(individual, subsets, universe_size=100):
    covered_elements = set()
    for i, bit in enumerate(individual):
        if bit == 1:
            covered_elements.update(subsets[i])
    coverage_score = len(covered_elements)
    
    subset_count = sum(individual)
    penalty = subset_count  
    fitness_score = coverage_score - penalty
    
    if coverage_score < universe_size:
        fitness_score -= (universe_size - coverage_score) * 10  # Penalizing uncovered elements
    
    return fitness_score

def select_parents(population, fitnesses):
    total_fitness = sum(fitnesses)
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

def genetic_algorithm(subsets, population_size=50, generations=50, mutation_rate=0.5, max_time=45):
    start_time = time.time()
    population = initialize_population(population_size, subsets)
    best_individual = max(population, key=lambda ind: fitness(ind, subsets))
    best_fitness_over_time = []
    for generation in range(generations):
        if time.time() - start_time >= max_time:
            break

        fitnesses = [fitness(ind, subsets) for ind in population]
        next_population = [] 
        for _ in range(population_size):
            parent1, parent2 = select_parents(population, fitnesses)
            child = crossover(parent1, parent2)
            let=random.random()
            if  let< mutation_rate:
                mutate(child)
            next_population.append(child)
            
        population = next_population
        current_best = max(population, key=lambda ind: fitness(ind, subsets))
        if fitness(current_best, subsets) > fitness(best_individual, subsets):
            best_individual = current_best
            
        best_fitness_over_time.append(fitness(best_individual, subsets))

    return best_individual, best_fitness_over_time


def run_experiments(sizes, num_experiments=10):
    mean_fitnesses = {}
    std_fitnesses = {}
    mean_fitness_over_generations = {size: [] for size in sizes}
    
    for size in sizes:
        all_fitnesses = []
        all_fitnesses_over_generations = [[] for _ in range(50)]
        
        for _ in range(num_experiments):
            scp = SetCoveringProblemCreator()
            subsets = scp.Create(usize=100, totalSets=size)
            best_solution, fitness_over_time = genetic_algorithm(subsets)
            all_fitnesses.append(fitness_over_time[-1])
            
            for gen in range(len(fitness_over_time)):
                all_fitnesses_over_generations[gen].append(fitness_over_time[gen])
        
        mean_fitnesses[size] = np.mean(all_fitnesses)
        std_fitnesses[size] = np.std(all_fitnesses)
        mean_fitness_over_generations[size] = [np.mean(fits) for fits in all_fitnesses_over_generations]
    
    return mean_fitnesses, std_fitnesses, mean_fitness_over_generations

def plot_results(mean_fitnesses, std_fitnesses, mean_fitness_over_generations):
    sizes = list(mean_fitnesses.keys())
    
    # Plot final fitness values
    plt.figure(figsize=(12, 6))
    plt.errorbar(sizes, [mean_fitnesses[size] for size in sizes],
                 yerr=[std_fitnesses[size] for size in sizes], fmt='o', capsize=5)
    plt.xlabel('Number of subsets')
    plt.ylabel('Best Fitness Value')
    plt.title('Mean and Standard Deviation of Final Fitness Value')
    plt.show()
    
    # Plot fitness over generations
    plt.figure(figsize=(12, 6))
    for size in sizes:
        plt.plot(range(len(mean_fitness_over_generations[size])), mean_fitness_over_generations[size], label=f'Size {size}')
    plt.xlabel('Generation')
    plt.ylabel('Mean Best Fitness Value')
    plt.title('Mean Best Fitness Value Over Generations')
    plt.legend()
    plt.show()


def main():
    print("Choose an option:")
    print("1. Individual Run")
    print("2. Batch Run")
    
    option = input("Enter your choice (1 or 2): ").strip()
    
    if option == "1":
        size = int(input("Enter the subset size (e.g., 50, 150, 250, 350): "))
        scp = SetCoveringProblemCreator()
        subsets = scp.Create(usize=100, totalSets=size)
        
        listOfSubsets = scp.ReadSetsFromJson("scp_test.json")
        best_solution, best_fitness_over_time = genetic_algorithm(listOfSubsets)
        print(f"Number of sets: {len(listOfSubsets)}/{len(subsets)}")
        print("Solution:", end=" ")
        for i, bit in enumerate(best_solution):
            print(f"{i}:{bit}", end=", ")
        print()
        print(f"Fitness value of best state: {fitness(best_solution, listOfSubsets)}")
        print(f"Minimum number of subsets that can cover the universe set: {sum(best_solution)}")
    
    elif option == "2":
        sizes = [50, 150, 250, 350]
        mean_fitnesses, std_fitnesses, mean_fitness_over_generations = run_experiments(sizes)
        plot_results(mean_fitnesses, std_fitnesses, mean_fitness_over_generations)
    
    else:
        print("Invalid option. Please enter 1 or 2.")
        sys.exit(1)

if __name__ == '__main__':
    main()
