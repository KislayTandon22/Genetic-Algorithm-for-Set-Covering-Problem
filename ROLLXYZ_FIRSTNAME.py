from SetCoveringProblemCreator import *
import random

def initialize_population(size, subsets):
    population = []
    for _ in range(size):
        individual = [random.choice([0, 1]) for _ in range(len(subsets))]
        population.append(individual)
    return population

def geneticAlgorithm(subsets, population_size=100, generations=100, mutation_rate=0.01):
    population = initialize_population(population_size, subsets)
    for generation in range(generation):
            




    return bestIndividual



def main():
    scp = SetCoveringProblemCreator()

#    ********* You can use the following two functions in your program

    subsets = scp.Create(usize=100,totalSets=200) # Creates a SetCoveringProblem with 200 subsets
    print(len(subsets))
    print()
    listOfSubsets = scp.ReadSetsFromJson("scp_test.json") #Your submission program should read from scp_test.json file and provide a good solution for the SetCoveringProblem.
    print(len(listOfSubsets))
    print()

    best_solution = geneticAlgorithm(listOfSubsets)
    print("Best solution found:", best_solution)
#    **********
#    Write your code for find the solution to the SCP given in scp_test.json file using Genetic algorithm.


if __name__=='__main__':
    main()