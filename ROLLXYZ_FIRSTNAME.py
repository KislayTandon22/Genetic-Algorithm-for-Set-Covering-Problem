from SetCoveringProblemCreator import *
import random
import time

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
        fitness_score -= (universe_size - coverage_score) * 10
    
    return fitness_score

def selection(population, fitnesses):
    selected = random.choices(population, weights=fitnesses, k=len(population))
    return selected

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]

def genetic_algorithm(subsets, population_size=100, generations=100, mutation_rate=0.01):
    population = initialize_population(population_size, subsets)
    for generation in range(generations):
        fitnesses = [fitness(ind, subsets) for ind in population]
        population = selection(population, fitnesses)
        next_generation = []
        for i in range(0, len(population), 2):
            parent1, parent2 = population[i], population[i+1]
            child1, child2 = crossover(parent1, parent2)
            mutate(child1, mutation_rate)
            mutate(child2, mutation_rate)
            next_generation.extend([child1, child2])
        population = next_generation
    best_individual = max(population, key=lambda ind: fitness(ind, subsets))
    return best_individual

def main():
    start_time = time.time()
    scp = SetCoveringProblemCreator()
    
    subsets = scp.Create(usize=100, totalSets=200)
    print(f"Number of subsets created: {len(subsets)}")
    
    listOfSubsets = scp.ReadSetsFromJson("scp_test.json")
    print(f"Number of subsets read from JSON: {len(listOfSubsets)}")
    
    best_solution = genetic_algorithm(listOfSubsets)
    
    
    
    print(f"Number of sets: {len(listOfSubsets)}/{len(subsets)}")
    
    
    print("Solution:", end=" ")
    for i, bit in enumerate(best_solution):
        print(f"{i}:{bit}", end=", ")

    print()
    print(f"Fitness value of best state: {fitness(best_solution, listOfSubsets)}")
    print(f"Minimum number of subsets that can cover the universe set: {sum(best_solution)}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.3f} seconds")

if __name__ == '__main__':
    main()
