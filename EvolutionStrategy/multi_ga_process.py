import argparse
import numpy as np
from .ga_utils import Individual, Problem, Variable

class NSGA2(object):

    """
    Note:
        Method:
           explore(main execution process)
               1. Initialize individual population
               2. compute objective function values from design variables and set information of individuals
               3. execute non dominated sort by using current population
               4. execute crowding sort in order to keep the variety of individuals
               5. apply evolution process(In this method, adapt the tournament method)
               6. adjust next individual population

        individual population is managed by individual class
    """

    def __init__(self, args):

        self.population_num = args.population_num
        self.generation_num = args.generation_num
        self.objectives_num = args.objectives_num
        self.design_variables_num = args.design_variables_num

        self.problem = Problem(args)

    def explore(self):

        # 1. create initialize population
        population = self.initialize_individual_population()

        for gene in range(self.generation_num):
            # 2. compute objective function and restore information into individual class
            population = self.set_information_of_population(population)
            # 3. execute non dominated sort
            all_fronts, pareto_front = self.conduct_non_dominated_sort(population)
            # 4. execute crowding sort
            population = self.conduct_crowding_sort(all_fronts)

            # decode design variables(optimization space into function space)
            design_variables_collection = self.problem.design_variables_class.convert_optimization_space_into_function_space(population)
            # 5. apply evolution process(crossover and mutation)
            next_design_variables_collection = self.apply_evolution_process(design_variables_collection)

            # 6. adjust next individual population
            population = self.adjust_next_population(next_design_variables_collection)

    def initialize_individual_population(self):
        population = []
        for num in range(self.population_num):
            # Initialize individual class
            individual = Individual()
            # set arguments in order to compute objective function
            for i in range(self.design_variables_num):
                norm_value = np.random.rand()
                variable_class = Variable(name, norm_value, min_value, max_value)

            # add individual class to population list
            population.append(individual)

        return population

    def set_information_of_population(self, population):
        for individual in population:
            # compute objective function values
            individual.objectives = self.problem.compute_objectives(individual)

        return population

    def conduct_non_dominated_sort(self, population):

        all_fronts = []  # fronts which can get by non dominated sort
        pareto_front = []  # pareto front

        return all_fronts, pareto_front

    def conduct_crowding_sort(self, all_fronts):
        population = []

        return population

    def apply_evolution_process(self, design_variables_collection):

        next_design_variables_collection = []

        return next_design_variables_collection

    def adjust_next_population(self, design_variable_collection):
        population = []


        return population


# Test function
def main():
    parser = argparse.ArgumentParser(description='NSGA2 parameters')
    parser.add_argument('--population_num', type=int, default=1000)
    parser.add_argument('--generation_num', type=int, default=5)
    parser.add_argument('--objective_num', type=int, default=2,
                        help='the number of objective functions [y1, y2, ...yn]')
    parser.add_argument('--design_variables_num', type=int, default=3,
                        help='the number of design variables [x1, x2, ... xm]')
    parser.add_argument('--bounds', type=str, default='0.5-1.5/3.5-4.5/2.5-5.5')
    parser.add_argument('--function_name', type=str, default='test', help='objective function name')
    parser.add_argument('--function_file_name', type=str, default='test_func',
                        help='objective function file path, please create function file under Function directory')
    args = parser.parse_args()

    nsga2 = NSGA2(args)



if __name__ == '__main__':
    main()