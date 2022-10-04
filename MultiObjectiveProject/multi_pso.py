import numpy as np
import random
from design_variable import DesignVariable
from multi_pso_utils import *
from integration_env_si import IntegrationEnvSwarm
from preprocess_for_integenv import PreprocessIntegrationEnv

# for test
from multiobjective_test_function import *

# PSO algorithm
# The main features of this algorithm is that once the local optimal solutions are determined, points of swarm intelligence restores
# So, if you want to finish the calculation and judge the better solution, you should use the hypervolume.
# In the case of Partial Swarm Optimization, the collections of design variables are defined as Swarm
# individual class
# 1. design vector
# 2. objectives
# 3. Personal best design variables
# 4. domination count
# 5. domination set S
# 6. velocity

# main class
# 0. Initialize the Swarm (create initial set of design variables and calculate the objectives)
# => start the learning
# Following operations are made to conduct every particle in Swarm
# 1. select the leader of design variables
# 2. update position of Swarm
# 3. operate mutation
# 4. By dealing with new collections of design variables, we have to calculate new objectives
# 5. confirm the dominance of the new point over the other points, if dominates, update personal best design variables
# => finish the each operation
# 6. update the new collections of design variables (This part includes the non dominated sort and crowding sort)
# if you need, you fix the shape of design variables into the computational environments


class MOPSO(object):

    def __init__(self, objective_indexes, objective_function_class, individual_num, epochs, optimal_dir, preprocess_env=None):
        self.name = 'PSO'
        # the name of lists of objective indexes
        self.objective_indexes = objective_indexes
        # the number of objective indexes
        self.objective_nums = len(self.objective_indexes)

        # determine the class for calculating the value of objective indexes
        self.objective_function_class = objective_function_class

        # the number of design variables
        self.design_variables_num = self.objective_function_class.design_variables_num

        # Build problem class
        self.problem_class = Problem(self.objective_nums, self.objective_function_class, optimal_dir, self.objective_indexes)

        # optimal direction
        self.optimal_dir = optimal_dir

        # the number of collections of design variables
        self.individual_num = individual_num
        # the number of learning times
        self.epochs = epochs

        # class for preparing for better collection of Swarm Intelligence
        self.preprocess_env = preprocess_env

        # the bounds of design variables (You have to use the normalized values in case of swarm
        # intelligence optimization)
        self.bounds = [0, 1]

        # the coefficients of updating Swarm positions
        self.inertia = 0.2
        self.cognitive_c1 = 0.2
        self.social_c2 = 0.2

    def create_initial_population(self):
        design_variables_collect = []

        for _ in range(self.individual_num):

            x = [np.random.rand() for _ in range(self.design_variables_num)]
            design_variables_collect.append(x)

        return design_variables_collect

    # 1. select the leader of collections
    def select_leader(self, non_dominated_design_variables_collect):
        """

        :param non_dominated_design_variables_collect: list type, the collections of only design variables
        :return:
        """

        return random.sample(non_dominated_design_variables_collect, 1)[0]

    # 2. update the Swarm points
    # maybe this process should operate real design space.
    def flight(self, individual, leader):

        # update the velocity of Particle of Swarm
        # the type of vel is numpy array ??
        vel = self.inertia * individual.velocity
        vel += self.cognitive_c1 * np.random.rand() * (individual.personal_best_vector - individual.normalized_design_vector)
        vel += self.social_c2 * np.random.rand() * (leader - individual.normalized_design_vector)

        # update the position
        new_position = individual.normalized_design_vector + vel

        for idx, d_value in enumerate(new_position):
            if d_value < self.bounds[0]:
                new_position[idx] = self.bounds[0]
            if d_value > self.bounds[1]:
                new_position[idx] = self.bounds[1]

        return new_position, vel

    # 3. mutation
    def mutate(self, design_variables):
        n_m = 20  # the range of this number is [20, 100]

        # helper function
        # change of the mutate
        def delta_l(target_num):

            return (2 * target_num) ** (1.0 / (1.0 + n_m)) - 1

        def delta_r(target_num):

            return 1.0 - (2 * (1.0 - target_num)) ** (1.0 / (1.0 + n_m))

        new_design_variables = []

        for dv in design_variables:
            target_num = np.random.rand()
            lower_num = self.bounds[0]
            upper_num = self.bounds[1]

            if target_num < 0.5:
                new_dv = dv + delta_l(target_num) * (dv - lower_num)
            else:
                new_dv = dv + delta_r(target_num) * (upper_num - dv)

            new_design_variables.append(new_dv)

        return new_design_variables

    # helper function for non dominated sort
    def set_population(self, design_variables_collect, thermal_design_variables_collect=None):
        population = []

        if thermal_design_variables_collect is None:
            thermal_design_variables_collect = [None for _ in range(self.individual_num)]

        for design_variables, therm_design_variables in zip(design_variables_collect, thermal_design_variables_collect):
            # define the individual class
            individual = Individual()
            individual.set_design_vector(design_variables)
            individual.therm_design_vector = therm_design_variables
            individual.velocity = np.zeros(self.design_variables_num)  # Initialize velocity list
            # calculate objectives
            meet_flag = self.problem_class.calculate_objectives(individual)
            # normalized design vector
            normalized_design_vector = self.objective_function_class.design_variable.norm(design_variables)
            individual.normalized_design_vector = np.array(normalized_design_vector)
            individual.personal_best_vector = np.array(normalized_design_vector)  # set the personal best design variables
            if meet_flag:
                population.append(individual)

        return population

    # non dominated sort
    def non_dominated_sort(self, population):
        pareto_fronts = []
        all_fronts = [[]]

        for individual in population:
            individual.domination_count = 0  # Initialize the number of dominance point
            individual.domination_set = []

            for other_individual in population:

                if self.problem_class.dominate(other_individual, individual):
                    individual.domination_set.append(other_individual)
                elif self.problem_class.dominate(individual, other_individual):
                    individual.domination_count += 1

            # add the pareto front
            if individual.domination_count == 0:
                pareto_fronts.append(individual)
                all_fronts[0].append(individual)

        idx = 0
        while len(all_fronts[-1]) > 0:
            rank_fronts = []

            for individual in all_fronts[idx]:
                for other_individual in individual.domination_set:
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
            # ToDo : Is it necessary for optimization ?
            self.problem_class.calculate_objectives(individual)
            # print('objectives:')
            # print(individual.objectives)
            # print('normalized objectives')
            # print(individual.normalized_objectives)

        for m in range(len(population_front[0].normalized_objectives)):
            # Initialize the edge of distance
            # shape of front => (index, individual class) tuple
            front = sorted(enumerate(population_front), key=lambda x: x[1].normalized_objectives[m])
            front[0][1].distances = np.inf
            front[-1][1].distances = np.inf

            for idx, val in enumerate(front[1:-1]):
                idx += 1
                front[idx][1].distances += (front[idx + 1][1].normalized_objectives[m] - front[idx - 1][1].normalized_objectives[m])
                crowding_distances[front[idx][0]] = front[idx][1].distances

        return crowding_distances

    def update_new_population(self, current_population, swarm):
        new_population = []

        # aggregate the collection
        aggregate_population = []
        aggregate_population.extend(current_population)
        aggregate_population.extend(swarm)
        # non dominated sort
        all_fronts, pareto_fronts = self.non_dominated_sort(aggregate_population)

        # restore previous generation's results
        previous_results = [all_fronts, pareto_fronts]

        print('length of all fronts:', len(all_fronts))
        print('the number of pareto front:', len(pareto_fronts))

        num = 0
        while len(new_population) + len(all_fronts[num]) < self.individual_num:
            new_population.extend(all_fronts[num])
            if len(all_fronts[num]) == 0:
                print('rank number:', num)
                break
            num += 1

        # redefine the current front
        current_front = all_fronts[num]
        # crowding sort
        # calculate crowding distances
        densities = self.calculate_crowding_distances(current_front)
        if self.optimal_dir == 'downleft':
            optimal_coef = 1.0
        else:
            optimal_coef = -1.0
        densities = sorted(enumerate(densities), key=lambda x: optimal_coef * x[1])
        # get the list of sorting indexes according to crowding distances
        densities = np.fromiter(map(lambda x: x[0], densities), dtype=int).tolist()

        # create new population
        elite_front = [current_front[idx] for idx in densities]
        new_population = np.concatenate((new_population, elite_front))

        return new_population[:self.individual_num], previous_results

    def explore(self, fixed_dict=None):
        # Initialize the Swarm (design variables in the swarm is the normalized state)
        if self.objective_function_class.name == 'test':
            current_design_variables_collect = self.create_initial_population()
            swarm = self.set_population(current_design_variables_collect)
        else:

            swarm_num = 0
            swarm = []
            while swarm_num < self.individual_num:
                # generate initial collection of swarm intelligence
                self.objective_function_class.design_variable.generate_si_design_variable_collect(self.individual_num * 5, fixed_dict)

                # Preprocess
                # change the design variables for evolutionary space into the thermal design variables
                self.objective_function_class.design_variable.generate_therm_design_variable_collect()
                # Choose the better collection
                current_thermal_design_variables_collect = self.objective_function_class.design_variable.therm_design_variable_collect
                after_preprocess_design_variables_collect = self.preprocess_env.select_better_individuals(current_thermal_design_variables_collect)
                # self.individual_num = len(after_preprocess_design_variables_collect)

                # replace the thermal design variables collection
                current_thermal_design_variables_collect = after_preprocess_design_variables_collect
                # reverse the design variables into the evolutionary space
                current_design_variables_collect = self.objective_function_class.design_variable.reverse_si_design_variables_collect(after_preprocess_design_variables_collect)

                target_swarm = self.set_population(current_design_variables_collect, current_thermal_design_variables_collect)

                swarm.extend(target_swarm)

                swarm_num = len(swarm)

                print('current number of swarm intelligence:', swarm_num)

        print('swarm length:', len(swarm))
        print(swarm[0].design_vector)
        print(swarm[0].objectives)

        # update the population (This process includes the non dominated sort and crowding sort)
        current_population, initial_results = self.update_new_population(swarm, swarm)

        # start the step of learning
        for epoch in range(self.epochs):
            # subtract the design variables from individual class
            non_dominated_design_variables_collect = []

            for cur_indiv in current_population:
                non_dominated_design_variables_collect.append(cur_indiv.design_vector)

            # all particles
            for individual in swarm:
                # select the leader
                leader = self.select_leader(non_dominated_design_variables_collect)
                # update the positions of Swarm intelligence positions
                new_position, velocity = self.flight(individual, leader)
                # execute the mutation process
                new_design_variables = self.mutate(individual.normalized_design_vector)
                # create the collection of new design variables(1)
                new_design_variables_collect = [new_design_variables]

                # create the collection of new thermal design variables
                if self.objective_function_class.name == 'Swarm':
                    self.objective_function_class.design_variable.si_design_variable_collect = new_design_variables_collect
                    self.objective_function_class.design_variable.generate_therm_design_variable_collect()
                    new_therm_design_variables_collect = self.objective_function_class.design_variable.therm_design_variable_collect
                else:
                    new_therm_design_variables_collect = None

                # determine the personal best design vector
                # set the new individual class
                new_individual_collect = self.set_population(new_design_variables_collect, new_therm_design_variables_collect)

                if len(new_individual_collect) == 0:
                    continue

                new_individual = new_individual_collect[0]

                # judge dominance
                if self.problem_class.dominate(new_individual, individual):

                    individual.personal_best_vector = new_individual.normalized_design_vector

                # update individual class
                individual = new_individual

            # update new population
            print('length of current population:', len(current_population), 'length of swarm:', len(swarm))
            current_population, previous_results = self.update_new_population(current_population, swarm)
            # draw each generation results
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            all_fronts, pareto_fronts = previous_results
            # subtract objectives
            # all individuals
            all_objectives = []
            for front in all_fronts:
                for individual in front:
                    all_objectives.append(individual.objectives)

            # pareto individuals
            pareto_objectives = []
            for individual in pareto_fronts:
                pareto_objectives.append(individual.objectives)

            print(all_objectives)
            print('')
            print('')
            print(pareto_objectives)

            # convert the numpy array
            all_objectives = np.array(all_objectives)
            pareto_objectives = np.array(pareto_objectives)

            if self.optimal_dir == 'upperright':
                all_objectives *= -1
                pareto_objectives *= -1

            if self.objective_nums == 2:
                # render
                plt.figure(figsize=(10, 6))
                plt.scatter(all_objectives[:, 0], all_objectives[:, 1], c='b', label='all')
                plt.scatter(pareto_objectives[:, 0], pareto_objectives[:, 1], c='r', label='pareto')
                plt.xlabel(self.objective_indexes[0])
                plt.ylabel(self.objective_indexes[1])
                plt.title('Evolution Epoch {}'.format(epoch))
                plt.legend()
                plt.show()

            elif self.objective_nums == 3:
                fig = plt.figure(figsize=(10, 6))
                ax = Axes3D(fig)

                ax.scatter(all_objectives[:, 0], all_objectives[:, 1], all_objectives[:, 2], c='b', label='all', alpha=0.1)
                ax.scatter(pareto_objectives[:, 0], pareto_objectives[:, 1], pareto_objectives[:, 2], c='r', label='pareto')
                ax.set_xlabel(self.objective_indexes[0])
                ax.set_ylabel(self.objective_indexes[1])
                ax.set_zlabel(self.objective_indexes[2])
                plt.title('Evolution Epoch {}'.format(epoch))
                ax.legend()
                plt.show()




# test code
def test_func_mo():
    # Local global variables
    # objective indexes
    objective_indexes = ['f1', 'f2']
    # the number of objective indexes
    objective_nums = len(objective_indexes)
    # the number of design variables
    design_variables_num = 4
    # optimal direction
    optimal_dir = 'upperright'
    # bounds
    bounds = [[0, 1], [0, 1], [0, 1], [0, 1]]
    # the number of the collections of Swarm Intelligence
    individual_num = 500
    # times of learning steps
    epochs = 10

    # build objective function class
    objective_function_class = ObjectTestFunc(objective_nums, design_variables_num, bounds)

    mopso = MOPSO(objective_indexes, objective_function_class, individual_num, epochs, optimal_dir)

    mopso.explore()

def test_mopso():

    # make integration env swarm class arguments
    # set the arguments of baseline aircraft and engine
    baseline_aircraft_name = 'A320'
    baseline_aircraft_type = 'normal'
    baseline_engine_name = 'V2500'
    baseline_propulsion_type = 'turbofan'

    # data path
    aircraft_data_path = './DataBase/aircraft_test.json'
    engine_data_path = './DataBase/engine_test.json'
    mission_data_path = './Missions/cargo1.0_passenger1.0_.json'

    # create database arguments
    data_base_args = [aircraft_data_path, engine_data_path, mission_data_path]

    baseline_args = [(baseline_aircraft_name, baseline_aircraft_type), (baseline_engine_name, baseline_propulsion_type),
                     (aircraft_data_path, engine_data_path)]

    # parameters at off design point (In most cases, it is at ground ot take-off)
    off_altitude = 0.0  # [m]
    off_mach = 0.0
    off_required_thrust = 133000  # [N]

    off_param_args = [off_altitude, off_mach, off_required_thrust]

    # make the current arguments
    current_aircraft_name = 'A320'
    current_aircraft_type = 'normal'
    current_engine_name = 'V2500'
    current_propulsion_type = 'turbofan'

    current_args = [(current_aircraft_name, current_aircraft_type), (current_engine_name, current_propulsion_type), (aircraft_data_path, engine_data_path)]

    # the position of engine equipping (2-dimensional vector)
    engine_mounting_positions = [0.2, 0.2]

    # the type of calculating the aerodynamic performance (In this case, lift by drag)
    ld_calc_type = 'constant-static'

    # constraint type
    constraint_type = 'TIT'

    # build the class for preprocess
    preprocess_env = PreprocessIntegrationEnv(current_aircraft_name, current_engine_name, current_aircraft_type, current_propulsion_type, data_base_args)

    # The global variables for optimization
    # the name's list of objective indexes
    objective_indexes = ['fuel_weight', 'engine_weight']

    # the number of individuals
    individual_num = 30
    # the number of times of learning
    epochs = 2

    # the direction of optimization
    optimal_dir = 'downleft'

    # build the objective function class
    objective_function_class = IntegrationEnvSwarm(baseline_args, current_args, off_param_args, mission_data_path, constraint_type, engine_mounting_positions, ld_calc_type)

    # class for multiobjective Particle Swarm Intelligence
    mopso = MOPSO(objective_indexes, objective_function_class, individual_num, epochs, optimal_dir, preprocess_env=preprocess_env)

    fixed_dict = {'BPR': 5}
    # run
    mopso.explore(fixed_dict)


if __name__ == '__main__':

    test_func_mo()

    # test_mopso()



