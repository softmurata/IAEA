import numpy as np
from .objective_function_wrapper import ObjectiveFunctionWrapper

class Variable(object):
    def __init__(self, name, norm_value, min_value, max_value):
        self.name = name
        self.value = None
        self.min = min_value
        self.max = max_value
        self.norm_value = norm_value

    def normalize(self):
        if self.norm_value is not None:
            self.norm_value = (self.value - self.min) / (self.max - self.min)

    def denormalize(self):
        if self.value is not None:
            self.value = self.min + self.norm_value * (self.max - self.min)


class Individual(object):
    """
    Note:
        Variables:
           design_variables
           objectives
           normalized_objectives
           dominates: Bool which indicates target individual dominates another individual
           dominated_count
           dominated_solutions: collection of solutions which target individual dominates
           rank: In case of non dominated sort, this information is necessary
           crowding distance: In case of crowding sort, this information is necessary

    """

    def __init__(self):
        self.design_variables = None
        self.objectives = None
        self.normalized_objectives = None
        self.dominates = None
        self.dominated_count = 0
        self.dominated_solutions = set()
        self.rank = None
        self.crowding_distance = None


class Problem(object):

    def __init__(self, args):
        # build objective function wrapper class
        self.objective_function_wrapper = ObjectiveFunctionWrapper()
        # take choice particular function class
        self.objective_function_wrapper.select_function_class(args)

        # redefine objective function class and design variable class
        self.objective_func_class = self.objective_function_wrapper.objective_func_class
        self.design_variables_class = self.objective_function_wrapper.design_variables_class

        self.objectives = np.zeros((1, args.objective_num))

        self.optimize_direction_vector = [1 for _ in range(args.objective_num)]

        if args.optimize_direction == 'upper_right':
            self.optimize_direction_vector = [-1 for _ in range(args.objective_num)]

    # judge dominance
    def dominates(self, individual2, individual1):
        objective1_values = individual1.objectives
        objective2_values = individual2.objectives

        # judge dominance
        non_dominated = all(map(lambda f: f[0] <= f[1], zip(objective1_values, objective2_values)))
        dominates = any(map(lambda f: f[0] < f[1], zip(objective1_values, objective2_values)))

        print()
        print('dominance information')
        print('non dominated:', non_dominated)
        print('dominates:', dominates)

        return non_dominated and dominates

    def compute_objectives(self, individual):
        individual.objectives = []  # correspond to multi objective functions
        individual.normalized_objectives = []

        objective_values = self.objective_function_wrapper.fitness(individual.design_variables)

        for idx, obj_f in enumerate(objective_values):
            optimize_dir = self.optimize_direction_vector[idx]

            obj_f *= optimize_dir

            if self.objectives.shape[0] == 1:
                self.objectives[0, idx] = obj_f

            # normalize objective function value
            mean = self.objectives[:, idx].mean()
            variance = self.objectives[:, idx].std()
            norm_obj_f = (obj_f - mean) / variance

            # add objectives and normalized objectives list
            individual.normalized_objectives.append(norm_obj_f)
            individual.objectives.append(obj_f)

        self.objectives = np.append(self.objectives, np.array([individual.objectives]), 0)

