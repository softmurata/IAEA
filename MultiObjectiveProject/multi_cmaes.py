import numpy as np
import random
from multi_cmaes_utils import *

# for test
from .multiobjective_test_function import *

# ToDo : platypus test

class MOCMAES(object):

    def __init__(self, objective_indexes, objective_func_class, individual_num, cmaes_coefs, diagonal_iterations, indicator='crowding', initial_search_point=None, check_consistency=False, epsilons=None):
        # the name list of objective indexes
        self.objective_indexes = objective_indexes
        # class for calculating the values of objective indexes
        self.objective_func_class = objective_func_class
        # the number of design variables
        self.design_variables_num = self.objective_func_class.design_variables_num
        # the number of collections of design variables for evolutionary space
        self.individual_num = individual_num

        # define the parameters of evolutionary strategy (s)
        self.cc, self.cs, self.damps, self.ccov, self.ccovsep, self.sigma = cmaes_coefs

        # the times of diagonalization
        self.diagonal_iterations = diagonal_iterations
        # other arguments
        self.indicator = indicator
        self.initial_search_point = initial_search_point
        self.check_consistency = check_consistency
        self.epsilons = epsilons

        # the collections of individual class
        self.population = []
        # the times of learning steps
        self.iterations = 0
        # the variable for memorizing eigen updates
        self.last_eigenupdate = 0

        # the evaluator for optimal solutions
        self.fitness_evaluator = None
        self.fitness_comparator = None

    def determine_constant(self):
        if self.sigma is None:
            self.sigma = 0.5

        if self.diagonal_iterations is None:
            self.diagonal_iterations = 150 * self.design_variables_num / self.individual_num

        self.diag_D = [1.0] * self.design_variables_num
        self.pc = [0.0] * self.design_variables_num
        self.ps = [0.0] * self.design_variables_num
        self.B = [[1.0 if i==j else 0.0 for j in range(self.design_variables_num)] for i in range(self.design_variables_num)]
        self.C = [[1.0 if i==j else 0.0 for j in range(self.design_variables_num)] for i in range(self.design_variables_num)]
        self.xmean = [0.0] * self.design_variables_num

        if self.initial_search_point is None:
            for i in range(self.design_variables_num):
                offset = self.sigma * self.diag_D[i]
                bound = self.objective_func_class.design_variable.bounds[i]
                min_value, max_value = bound
                rangev = max_value - min_value - 2 * self.sigma * self.diag_D[i]

                if offset > 0.4 * (max_value - min_value):
                    offset = 0.4 * (max_value - min_value)
                    rangev = 0.2 * (max_value - min_value)

                self.xmean[i] = min_value + offset + random.uniform(0.0, 1.0) * rangev

        else:
            for i in range(self.design_variables_num):
                self.xmean[i] = self.initial_search_point[i] + self.sigma * self.diag_D[i] * random.gauss()

        # coefficient
        self.chi_N = np.sqrt(self.design_variables_num) * (1.0 - 1.0 / (4.0 * self.design_variables_num) + 1.0 / (21.0 * self.design_variables_num ** 2))
        self.mu = int(np.floor(self.individual_num / 2))
        self.weights = [np.log(self.mu + 1.0) - np.log(i + 1.0) for i in range(self.mu)]

        # normalization of weights
        sum_weights = sum([w for w in self.weights])
        self.weights = [w / sum_weights for w in self.weights]

        sum_sq_weights = sum([w ** 2 for w in self.weights])
        self.mueff = 1.0 / sum_sq_weights

        # parameters for strategy
        if self.cs is None:
            self.cs = (self.mueff + 2.0) / (self.design_variables_num + self.mueff + 3.0)

        if self.damps is None:
            self.damps = (1.0 + 2.0 * max(0, np.sqrt((self.mueff - 1.0) / (self.design_variables_num + 1.0)) - 1.0)) + self.cs

        if self.cc is None:
            self.cc = 4.0 / (self.design_variables_num + 4.0)

        if self.ccov is None:
            self.ccov = 2.0 / (self.design_variables_num + np.sqrt(2)) / (self.design_variables_num + np.sqrt(2)) / self.mueff + (1.0 - (1.0 / self.mueff)) * min(1.0 , (2.0 * self.mueff - 1.0) / (self.mueff + (self.design_variables_num + 2.0) ** 2 ))

        if self.ccovsep is None:
            self.ccovsep = min(1.0, self.ccov * (self.design_variables_num + 1.5) / 3.0)

    # decomposition of covariance array
    def decompose_covariance_arr(self):

        self.last_eigenupdate = self.iterations

        # if iteration == 0
        if self.diagonal_iterations >= self.iterations:
            for i in range(self.design_variables_num):
                self.diag_D[i] = np.sqrt(self.C[i][i])

        else:
            for i in range(self.design_variables_num):
                for j in range(i + 1):
                    self.B[i][j] = self.B[j][i] = self.C[i][j]

            offdiag = [0.0] * self.design_variables_num
            tred2(self.design_variables_num, self.B, self.diag_D, offdiag)
            tql2(self.design_variables_num, self.diag_D, offdiag, self.B)

            # error process
            for i in range(self.design_variables_num):
                if self.diag_D[i] < 0.0:
                    self.diag_D[i] = 0.0

                self.diag_D[i] = np.sqrt(self.diag_D[i])

    def create_init_population(self):
        population = []
        for _ in range(self.individual_num):
            individual = Individual()

            while True:
                # flag which target solution is feasible in this problem
                feasible = True

                individual.design_vector = [0 for _ in range(self.design_variables_num)]

                for i in range(self.design_variables_num):
                    value = self.xmean[i] + self.sigma * self.diag_D[i] * random.gauss(0.0, 1.0)
                    # boundary condition
                    bound = self.objective_func_class.design_variable.bounds[i]
                    min_value, max_value = bound

                    if value < min_value or value > max_value:
                        feasible = False
                        break

                    individual.design_vector[i] = value

                if feasible:
                    break

            population.append(individual)
        # update times of learning steps
        self.iterations += 1
        return population

    def mutate(self):

        population = []

        for _ in range(self.individual_num):
            artmp = [0.0] * self.design_variables_num

            while True:
                feasible = True
                # build the class of individual
                individual = Individual()

                individual.design_vector = [0 for _ in range(self.design_variables_num)]

                for i in range(self.design_variables_num):
                    artmp[i] = self.diag_D[i] * random.gauss(0.0, 1.0)

                for i in range(self.design_variables_num):
                    mutation = 0.0

                    for k in range(self.design_variables_num):
                        mutation += self.B[i][k] * artmp[k]

                    value = self.xmean[i] + self.sigma * mutation

                    # boundary condition
                    bound = self.objective_func_class.design_variable.bounds[i]
                    min_value, max_value = bound

                    if value < min_value or value > max_value:
                        feasible = False
                        break

                    individual.design_vector[i] = value

                if feasible:
                    break

            population.append(individual)

        # update times of learning steps
        self.iterations += 1
        return population

    # update coefficient
    def update_coefficient(self):
        # previous design vector
        xold = self.xmean[:]
        BDz = [0.0] * self.design_variables_num
        artmp = [0.0] * self.design_variables_num

    def explore(self):
        # Initialize the
        self.determine_constant()
        # generate initial population (if epoch = 0)
        # mutate (if epoch > 0)




"""
class MOCMAES(object):

    def __init__(self, objective_indexes, objective_func_class, individual_num, epochs, initial_covariances, optimal_dir, preprocess_env_class=None):

        # the name list of objective indexes
        self.objective_indexes = objective_indexes
        # the number of objective indexes
        self.objective_nums = len(self.objective_indexes)
        # set class for calculating the values of objective indexes
        self.objective_func_class = objective_func_class

        self.individual_num = individual_num  # the number of the collection of design variables
        self.epochs = epochs  # the number of times of learning

        # the number of design variables
        self.design_variables_num = self.objective_func_class.design_variables_num

        # the direction for optimization
        self.optimal_dir = optimal_dir

        # Build problem class
        self.problem_class = Problem(self.individual_num, self.objective_nums, self.objective_func_class, self.optimal_dir, self.objective_indexes)

        # Build the class for preprocess
        self.preprocess_env_class = preprocess_env_class

        # covariances
        self.init_sigma, self.init_mean = initial_covariances
        self.init_mean = np.zeros(self.design_variables_num)

    # create initial collections of design variables for test function
    def create_init_population(self):
        bounds = [0, 1]
        design_variables_collect = []

        for _ in range(self.individual_num):
            x = np.random.rand(self.design_variables_num)
            design_variables_collect.append(x)

        return design_variables_collect

    def set_population(self, design_variables_collect, thermal_design_variables_collect=None):
        population = []
        if thermal_design_variables_collect is None:
            thermal_design_variables_collect = [None for _ in range(self.individual_num)]

        for design_variables, therm_design_variable in zip(design_variables_collect, thermal_design_variables_collect):
            individual = Individual()
            individual.design_vector = design_variables
            individual.therm_design_vector = therm_design_variable
            # Initialize
            if individual.cov is None:
                individual.cov = np.identity(self.design_variables_num)
                individual.sigma = self.init_sigma
                individual.pSucc = self.problem_class.ptarget
                individual.pEvol = 0

            meet_flag = self.problem_class.calculate_objectives(individual)

            if meet_flag:
                population.append(individual)

        return population

    # non dominated sort
    def non_dominated_sort(self, population):
        # Initialize the front list
        all_fronts = [[]]
        pareto_fronts = []

        for individual in population:
            individual.domination_count = 0
            individual.dominated_solutions = set()

            for other_individual in population:

                if self.problem_class.dominate(other_individual, individual):
                    individual.dominated_solutions.add(other_individual)
                elif self.problem_class.dominate(individual, other_individual):
                    individual.domination_count += 1

            # add the pareto front
            if individual.domination_count == 0:
                pareto_fronts.append(individual)
                all_fronts[0].append(individual)

        # calculate pareto rank
        idx = 0
        while len(all_fronts[-1]) > 0:
            rank_fronts = []

            for individual in all_fronts[idx]:
                for other_individual in individual.dominated_solutions:
                    # diminish the number of dominance points
                    other_individual.domination_count -= 1

                    if other_individual.domination_count == 0:
                        rank_fronts.append(other_individual)

            idx += 1
            all_fronts.append(rank_fronts)

        return all_fronts, pareto_fronts

    # calculate the crowding distances
    def calculate_crowding_distances(self, population_front):
        crowding_distances = [0 for _ in range(len(population_front))]

        for individual in population_front:
            individual.distances = 0
            self.problem_class.calculate_objectives(individual)

        for m in range(len(population_front[0].normalized_objectives)):
            # Initialize the edge of distance by infinity
            front = sorted(enumerate(population_front), key=lambda x: x[1].normalized_objectives)
            front[0][1].distances = np.inf
            front[-1][1].distances = np.inf

            for idx, dis in enumerate(front[1:-1]):
                idx += 1
                front[idx][1].distances += (front[idx + 1][1].normalized_objectives[m] - front[idx - 1][1].normalized_objectives[m])
                crowding_distances[front[idx][0]] = front[idx][1].distances

        return crowding_distances

    def select_best_individual(self, population):
        next_population = []
        all_fronts, pareto_fronts = self.non_dominated_sort(population)

        previous_results = [all_fronts, pareto_fronts]

        num = 0
        while len(next_population) + len(all_fronts[num]) < self.individual_num:
            next_population.extend(all_fronts[num])
            if len(all_fronts[num]) == 0:
                break
            num += 1

        # redefine the current front (for crowding sort)
        current_front = all_fronts[num]
        # crowding sort
        crowding_distances = self.calculate_crowding_distances(current_front)

        if self.optimal_dir == 'downleft':
            optimal_coef = 1.0
        else:
            optimal_coef = -1.0

        crowding_distances = sorted(enumerate(crowding_distances), key=lambda x: optimal_coef * x[1])
        # get the list of sorting indexes according to crowding distances
        crowding_distances_indexes = np.fromiter(map(lambda x: x[0], crowding_distances), dtype=int).tolist()

        # create elite population
        elite_population = [current_front[idx] for idx in crowding_distances_indexes]
        next_population = np.concatenate((next_population, elite_population))

        return next_population[:self.individual_num], previous_results

    def explore(self, fixed_dict=None):

        # generate initial collections
        current_design_variables_collect = self.create_init_population()

        if self.objective_func_class.name == 'Swarm':
            self.objective_func_class.design_variable.si_design_variable_collect = current_design_variables_collect
            # change the thermal design variables
            self.objective_func_class.design_variable.generate_therm_design_variable_collect()
            current_thermal_design_variable_collect = self.objective_func_class.design_variable.therm_design_variable_collect

            # preprocess
            after_preprocess_design_variable_collect = self.preprocess_env_class.select_better_individuals(current_thermal_design_variable_collect)

            # reverse the thermal design variables into the evolutionary space
            current_design_variables_collect = self.objective_func_class.design_variable.reverse_si_design_variales_collect(after_preprocess_design_variable_collect)
            current_thermal_design_variables_collect = after_preprocess_design_variable_collect

        else:
            current_thermal_design_variables_collect = None

        # set something
        current_population = self.set_population(current_design_variables_collect, current_thermal_design_variables_collect)

        for epoch in range(self.epochs):

            # step 1: reproduction
            Q = []
            for individual in current_population:
                new_individual = self.problem_class.mutate(individual)
                Q.append(new_individual)

            # print('length of Q :', len(Q))
            # step 2: updates
            for idx in range(self.individual_num):
                # update step size
                self.problem_class.update_step_size(current_population[idx])
                self.problem_class.update_step_size(Q[idx])

                # update covariance matrix
                self.problem_class.update_covariance(Q[idx])

            Q.extend(current_population)

            # step 3: selection
            # non dominated sort
            # crowding distance
            current_population, previous_results = self.select_best_individual(Q)
            # print('current population length:', len(current_population))

            # draw the results
            from mpl_toolkits.mplot3d import Axes3D
            import matplotlib.pyplot as plt

            all_fronts, pareto_fronts = previous_results

            all_objective_data = []
            for front in all_fronts:
                for individual in front:
                    all_objective_data.append(individual.objectives)

            pareto_objective_data = []
            for individual in pareto_fronts:
                pareto_objective_data.append(individual.objectives)

            all_objective_data = np.array(all_objective_data)
            pareto_objective_data = np.array(pareto_objective_data)

            if self.optimal_dir == 'upperright':
                all_objective_data *= -1
                pareto_objective_data *= -1

            if self.objective_nums == 3:
                fig = plt.figure(figsize=(10, 6))
                ax = Axes3D(fig)

                ax.scatter(all_objective_data[:, 0], all_objective_data[:, 1], all_objective_data[:, 2], c='b', label='all', alpha=0.1)
                ax.scatter(pareto_objective_data[:, 0], pareto_objective_data[:, 1], pareto_objective_data[:, 2], c='r', label='pareto')

                ax.set_xlabel(self.objective_indexes[0])
                ax.set_ylabel(self.objective_indexes[1])
                ax.set_zlabel(self.objective_indexes[2])

                ax.legend()
                plt.title('Evolution Epoch {}'.format(epoch))
                plt.show()

            elif self.objective_nums == 2:
                plt.figure(figsize=(10, 6))
                plt.scatter(all_objective_data[:, 0], all_objective_data[:, 1], c='b', label='all')
                plt.scatter(pareto_objective_data[:, 0], pareto_objective_data[:, 1], c='r', label='pareto')

                plt.xlabel(self.objective_indexes[0])
                plt.ylabel(self.objective_indexes[1])
                plt.title('Evolution Epoch {}'.format(epoch))
                plt.legend()
                plt.show()


# test function
def test_func_mo():
    # the name list of objective indexes
    objective_indexes = ['f1', 'f2']
    # the number of objective indexes
    objective_nums = len(objective_indexes)
    # the number of design variables
    design_variables_num = 4
    # bounds
    bounds = [[0, 1], [0, 1], [0, 1], [0, 1]]


    # build objective function class
    objective_func_class = ObjectTestFunc(objective_nums, design_variables_num, bounds)
    # build preprocess class
    preprocess_env_class = None

    # Global variables for optimization
    individual_num = 400
    epochs = 10
    # initial covariances
    init_sigma = 2.0
    init_mean = 0.0
    initial_covariances = [init_sigma, init_mean]

    # optimal direction
    optimal_dir = 'upperright'

    # build optimization class
    mocmaes = MOCMAES(objective_indexes, objective_func_class, individual_num, epochs, initial_covariances, optimal_dir, preprocess_env_class)

    # run
    fixed_dict = None
    mocmaes.explore(fixed_dict)

if __name__ == '__main__':
    test_func_mo()
    
"""








