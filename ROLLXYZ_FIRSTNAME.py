from SetCoveringProblemCreator import *
        
def main():
    scp = SetCoveringProblemCreator()

#    ********* You can use the following two functions in your program

    subsets = scp.Create(usize=100,totalSets=200) # Creates a SetCoveringProblem with 200 subsets
    print(len(subsets))
    print()
    listOfSubsets = scp.ReadSetsFromJson("scp_test.json") #Your submission program should read from scp_test.json file and provide a good solution for the SetCoveringProblem.
    print(len(listOfSubsets))
    print()
    
#    **********
#    Write your code for find the solution to the SCP given in scp_test.json file using Genetic algorithm.


if __name__=='__main__':
    main()