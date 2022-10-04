import numpy as np


# Keep the indivdual data class
class Individual(object):

    def __init__(self):
        self.rank = None
        self.crowding_distance = 0
        self.objectives = None  # objective values
        self.normalized_objectives = None  # normalize objective values
        self.other_objective_values = None
        self.features = None  # design variables for swarm space
        self.therm_features = None
        self.dominates = None
        self.dominated_solutions = set()
        self.domination_count = 0

    def set_objectives(self, objectives):
        self.objectives = objectives


# Define problem
class Problem(object):

    def __init__(self, objective_nums, objective_func_class, optimal_dir, objective_indexes=None):

        # set the objective function class
        self.objective_func_class = objective_func_class
        # the number of objective functions
        self.objective_nums = objective_nums
        # objectives
        self.objectives = np.zeros((1, self.objective_nums))
        # optimization direction (default)
        self.optimal_dir_vec = [1 for _ in range(self.objective_nums)]
        # names of objectives
        self.objective_indexes = objective_indexes

        if optimal_dir == 'upperright':
            self.optimal_dir_vec = [-1 for _ in range(self.objective_nums)]

    # judge dominance
    def dominate(self, individual2, individual1):
        """
        individual class have already calculated and inserted values
        :param individual2:
        :param individual1: (Initial all design variables collection)
        :return:
        """
        
        objective1_values = individual1.objectives
        objective2_values = individual2.objectives
        # objective1_values = individual1.normalized_objectives
        # objective2_values = individual2.normalized_objectives

        non_dominated = all(map(lambda f: f[0] <= f[1], zip(objective1_values, objective2_values)))
        dominates = any(map(lambda f: f[0] < f[1], zip(objective1_values, objective2_values)))

        return non_dominated and dominates

    def calculate_objectives(self, individual):
        # initialize list
        individual.objectives = []
        individual.normalized_objectives = []

        # create function values
        # In the objective function class, method called fitness have to be created
        if self.objective_func_class.name == 'test':
            objective_values = self.objective_func_class.fitness(individual.features)
            other_objective_values = None
        else:
            objective_values_dict = self.objective_func_class.fitness(individual.therm_features)

            if objective_values_dict is None:
                return False, None
            # create list of objective values
            objective_values = []
            other_objective_values = []
            for key, value in objective_values_dict.items():
                if key in self.objective_indexes:
                    objective_values.append(value)
                else:
                    other_objective_values.append(value)

        if objective_values is None:

            return False, None

        for idx, fval in enumerate(objective_values):

            optim_dir_coef = self.optimal_dir_vec[idx]

            fval *= optim_dir_coef  # if you want to change optimization direction, you have to multiply minus to the objectives

            if self.objectives.shape[0] == 1:
                self.objectives[0, idx] = fval

            # standard normalization
            # calculate mean and variance
            mean = self.objectives[:, idx].mean()
            var = self.objectives[:, idx].std()
            norm_obj = (fval - mean) / var

            individual.normalized_objectives.append(norm_obj)
            individual.objectives.append(fval)

        self.objectives = np.append(self.objectives, np.array([individual.objectives]), 0)

        return True, other_objective_values


