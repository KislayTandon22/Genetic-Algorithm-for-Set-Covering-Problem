import random
import time
from SetCoveringProblemCreator import *

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

def genetic_algorithm(subsets, population_size=100, generations=100, mutation_rate=0.01, start_time=None, max_time=20):
    population = initialize_population(population_size, subsets)
    best_individual = max(population, key=lambda ind: fitness(ind, subsets))

    for generation in range(generations):
        fitnesses = [fitness(ind, subsets) for ind in population]
        population = selection(population, fitnesses)
        
        next_generation = []
        for i in range(0, len(population), 1):
            parent1, parent2 = random.sample(population, 2)
            child1, child2 = crossover(parent1, parent2)
            
            mutate(child1, mutation_rate)
            

            mutate(child2, mutation_rate)
            # Check time and update best individual after second mutation
            if time.time() - start_time > max_time:
                return max([child2, best_individual], key=lambda ind: fitness(ind, subsets))
            if fitness(child2, subsets) > fitness(best_individual, subsets):
                best_individual = child2

            next_generation.extend([child1, child2])

        population = next_generation

    return best_individual

def main():
    start_time = time.time()
    scp = SetCoveringProblemCreator()
    
    subsets = scp.Create(usize=100, totalSets=200)
    print(f"Number of subsets created: {len(subsets)}")
    
    listOfSubsets = scp.ReadSetsFromJson("scp_test.json")
    print(f"Number of subsets read from JSON: {len(listOfSubsets)}")
    
    best_solution = genetic_algorithm(listOfSubsets, start_time=start_time)
    
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
