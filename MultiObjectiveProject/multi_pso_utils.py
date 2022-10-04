import numpy as np

# class for Partial Swarm Optimization
# This class plays the part in keeping the required data for non-dominated sort and crowding sort
class Individual(object):

    def __init__(self):
        self.design_vector = None  # vector of design variables(list)
        self.normalized_design_vector = None  # Normalization of design vector
        self.objectives = None  # the values of objectives(list)
        self.normalized_objectives = None
        self.personal_best_vector = None  # Pbest (normalized)
        self.distances = None  # distance for crowding sort
        self.domination_count = 0
        self.domination_set = None  # the collection of the design variables which concentrated point dominates
        self.velocity = None

    def set_objectives(self, objectives):
        self.objectives = objectives

    def set_design_vector(self, design_variables):
        self.design_vector = np.array(design_variables)


class Problem(object):

    def __init__(self, objective_nums, objective_func_class, optimal_dir, objective_indexes=None):

        # class for calculating objective indexes
        self.objective_func_class = objective_func_class
        # the number of objective indexes
        self.objective_nums = objective_nums
        # objectives
        self.objectives = np.zeros((1, self.objective_nums))
        # optimal direction (mainly 'downleft' or 'upperright')
        self.optimal_dir = optimal_dir
        # list of names for objective indexes (ex. 'fuel_weight',..)
        # no need for test function
        self.objective_indexes = objective_indexes
        # According to the optimal direction, we have to choose the direction vector for optimization
        if self.optimal_dir == 'downleft':
            self.optimal_dir_vec = [1.0 for _ in range(self.objective_nums)]
        else:
            self.optimal_dir_vec = [-1.0 for _ in range(self.objective_nums)]

    # judge dominance
    def dominate(self, individual2, individual1):

        objective1_values = individual1.objectives
        objectives2_values = individual2.objectives

        non_dominated = all(map(lambda f: f[0] < f[1], zip(objective1_values, objectives2_values)))
        dominates = any(map(lambda f: f[0] < f[1], zip(objective1_values, objectives2_values)))

        return non_dominated and dominates

    # calculate objectives
    def calculate_objectives(self, individual):
        # Initialize list
        individual.objectives = []
        individual.normalized_objectives = []

        # create function values
        if self.objective_func_class.name == 'test':
            objective_values = self.objective_func_class.fitness(individual.design_vector)
        else:
            # if you run the main part, you have to change the design variables for evolutionary space
            # into the thermal design variables
            objective_dict = self.objective_func_class.fitness(individual.therm_design_vector)

            # if the result of objective indexes is None, the value of returns is boolean
            if objective_dict is None:
                return False

            # create the list of objective values
            objective_values = []
            for key, value in objective_dict.items():
                if key in self.objective_indexes:
                    objective_values.append(value)

        if objective_values is None:

            return False

        for idx, obj in enumerate(objective_values):

            # the coefficient for optimal direction
            optimal_dir_coef = self.optimal_dir_vec[idx]

            obj *= optimal_dir_coef

            if self.objectives.shape[0] == 1:
                self.objectives[0, idx] = obj

            # calculate mean and variances
            mean = self.objectives[:, idx].mean()
            var = self.objectives[:, idx].std() or 1

            norm_obj = (obj - mean) / var

            individual.normalized_objectives.append(norm_obj)
            individual.objectives.append(obj)

        self.objectives = np.append(self.objectives, np.array([individual.objectives]), 0)

        return True




