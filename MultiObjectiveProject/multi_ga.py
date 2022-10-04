import numpy as np
import time
import os
import functools
from integration_env_si import IntegrationEnvSwarm
from multi_ga_utils import *
from preprocess_for_integenv import PreprocessIntegrationEnv

# for test
from multiobjective_test_function import *

# ToDo Elite selection cannot work well, so we have to repair the part of function called Evolutionary process
# In addition, method of extracting better pareto solutions should be devised in this weekend

class NSGA2(object):
    """
    Note:

        In this part, i have to describe the configuration of algorithm

    Attributes
    ------------------

    objective_indexes: list => [str, str, ...]
                       the list of names of objective indexes
    objective_nums: int
                    the number of objective indexes
    names_of_objective_indexes: dict
                                type of dictionary which contains the combination of the names of objective indexes
                                and index number
    objective_function_class: class object
                              class object which calculating and keep the results of thermal performances
    individual_num: int
                    the number of initial individual
    epochs: int
            the number of learning steps or generation
    design_variables_num: int
            the number of design variables [x1, x2, x3,...]
    optimal_dir: str
                 the optimal direction, for example, if you want to choose original point as the optimal direction,
                 you have to set 'downleft' as the optimal_dir
    problem_class: class object
                   class object for converting the values of objective indexes into the values in the optimization space
                   and includes the various operations on the optimization
    crossover_times: int
                     the number of the natural operation called crossover
    individual_mutate_rate: float
                            the ratio of mutation for one individual against population
    design_variable_mutate_rate: float
                                 the ratio of mutation for one value in the design variables
    preprocess_env_class: class object
                          class object for preparing the better individual sets such as surviving
                          the collection meeting the requirements of optimization
    survival_coef: float
                   the ratio of surviving the collection against the overall individual collection
                   in the process of crowding sort
    generation_num: int
                    the initial number of sets of the individuals at each generation
    elite_sort_objective_names: str
                                the name of objectives for extracting the elite individuals

    """

    def __init__(self, objective_indexes, objective_function_class, individual_num, epochs, optimal_dir='downleft',
                 preprocess_env_class=None, crossover_times=10, individual_mutate_rate=0.1,
                 design_variables_mutate_rate=0.1, survival_coef=0.6, decline_step=0, gene_step=5,
                 elite_sort_objective_names='euclid distance'):

        # the name lists of objective indexes (dictionary :{objective_function_name: optimization type)
        self.objective_indexes = objective_indexes
        self.objective_nums = len(objective_indexes)
        # if you want to survey more objective indexes, you have to add this index name and its index to this dictionary
        self.names_of_objective_indexes = {'fuel_weight': 0, 'engine_weight': 1, 'aircraft_weight': 2,
                                        'max_takeoff_weight': 3, 'electric_weight': 4, 'isp': 5,
                                        'sfc': 6, 'co_blade_h': 7, 'stage_numbers': 8, 'th_ew_ratio': 9}
        # objective constant
        self.objective_constant = 1000
        # class for calculating objective indexes (Already built)
        self.objective_function_class = objective_function_class

        self.individual_num = individual_num  # the number of design variables
        self.epochs = epochs  # the number of learning process

        # design variables num
        self.design_variables_num = self.objective_function_class.design_variables_num

        # optimization direction
        self.optimal_dir = optimal_dir  # type(str)

        # problem class
        self.problem_class = Problem(self.objective_nums, self.objective_function_class, self.optimal_dir, self.objective_indexes)

        # the number of crossover
        self.crossover_times = crossover_times
        # probability of mutating for individuals
        self.individual_mutate_rate = individual_mutate_rate
        # probability of mutating for design variables
        self.design_variable_mutate_rate = design_variables_mutate_rate

        # set the preprocess class
        self.preprocess_env_class = preprocess_env_class

        # the reference point
        # self.reference_point = [0 for _ in range(self.objective_nums)]

        # The problem: 1. More than 0.5, elite selection does not work well.
        self.survival_coef = survival_coef  # survival ratio for crowding sort

        self.generation_num = self.crossover_times * 2

        self.decline_step = decline_step

        self.gene_step = gene_step

        self.elite_sort_objective_names = elite_sort_objective_names  # others are 'fuelburn' and 'euclid distance'

    # control function (generation number, survival coefficient for crowding sort and save initial population)
    def control_generation_num(self, epoch):

        """

        :param epoch: current epoch
        :return: None
        """

        if epoch == 0 and self.generation_num >= 1000:
            self.generation_num = int(self.generation_num * 0.8)
        else:
            self.generation_num -= self.decline_step

        # set minimum value
        if self.generation_num <= 30:
            self.generation_num = 30

    def control_survival_coef(self, epoch, diff=0):

        """

        :param epoch: current epoch
        :param diff: the degree of change
        :return: None
        """

        if epoch >= int(self.epochs * 0.3) - 1:
            self.survival_coef -= diff * epoch

    def control_mutation_rate(self, epoch, diff=0.1):
        """

        :param epoch: current epoch
        :param diff: the degree of change
        :return: None
        """
        if epoch >= int(self.epochs * 0.3) - 1:
            self.individual_mutate_rate += diff
            self.design_variable_mutate_rate += diff

    # save initial collections of design variables as npy file
    def save_initial_collect(self, population, file_number):
        """

        :param population:
        :param file_number:
        :return: None
        """
        # set file path
        constraint_type = self.objective_function_class.constraint_type
        save_filename = './GA_results/InitialIndividual/{0}/{1}.npy'.format(constraint_type, file_number)

        # construct design variables array
        dvs = []
        for individual in population:
            dvs.append(individual.therm_features)
        dvs = np.array(dvs)

        np.save(save_filename, dvs)

    # create initial design variable collections
    def create_init_population(self):
        bounds = [0, 1]
        design_variables_collect = []

        for _ in range(self.individual_num):
            x = [np.random.rand() for _ in range(self.design_variables_num)]
            design_variables_collect.append(x)

        return design_variables_collect

    # create suitable objective values
    def change_suitable_form_of_objectives(self, objective_func_values):
        """

        :param objective_func_values:
        :return: suitable_obj_values
        """
        # optimization direction
        # ['downleft', 'downright', 'upperleft', 'upperright']
        # attention
        # set the priority of objective indexes and insert these indexes in right order

        # Initialize objective function values list
        suitable_obj_values = []

        for objective_name, optim_direct in self.objective_indexes.items():
            target_idx = self.names_of_objective_indexes[objective_name]
            # subtract the objective values list from the previous results
            target_objective_lists = objective_func_values[target_idx]
            # according to the optimization direction, change the values
            if optim_direct == 0:
                suitable_obj_values.append(target_objective_lists)
            else:
                target_objective_lists = [self.objective_constant / obj_val for obj_val in target_objective_lists]
                suitable_obj_values.append(target_objective_lists)

        suitable_obj_values = np.array(suitable_obj_values).T.tolist()

        return suitable_obj_values

    def add_sample(self, design_variables_collect):
        dv_sets = np.array(design_variables_collect)
        bounds = []
        for idx in range(self.design_variables_num):
            each_dv = dv_sets[:, idx]
            min_dv, max_dv = np.min(each_dv), np.max(each_dv)
            bounds.append([min_dv, max_dv])

        new_dv_sets = []
        for _ in range(self.crossover_times):
            new_dv = [bounds[idx][0] + np.random.rand() * (bounds[idx][1] - bounds[idx][0]) for idx in range(self.design_variables_num)]
            new_dv_sets.append(new_dv)

        new_dv_sets.extend(design_variables_collect)

        return new_dv_sets

    def explore(self, fixed_dict=None, result_dirs=None, initial_explore_type=False):
        """

        :param fixed_dict: type of dictionary which has the combination of names of fixed design variable and
                           values of it

        :param result_dirs: the list of directory names for restoring the computational results

        :param initial_explore_type: the boolean which indicates whether or not dealing with previous results
                                     as the initial individual collection
        :return: None


        Attributes
        ----------------


        fixed_dict: dict

        result_dirs: list

        initial_explore_type: boolean


        Note:

            Algorithm:


            1. create initial individuals

            2. loop 2.1-2.6 at epoch times

              2.1. calculate objectives on the constraint condition + select determined number of individuals

              2.2. operate non dominated sort + crowding sort

              2.3. restore each step of pareto solutions and all solutions

              2.4. operate evolutionary process(crossover, mutation)

              2.5. update the collection of individuals

        """
        # generate initial swarm intelligence design variables collection
        if self.objective_function_class.name == 'Swarm':
            if initial_explore_type:
                if self.objective_function_class.constraint_type == 'TIT':
                    initial_path = './GA_results/InitialIndividual/TIT/0.npy'
                else:
                    initial_path = './GA_results/InitialIndividual/COT/0.npy'

                current_thermal_design_variables_collect = np.load(initial_path).tolist()
                current_design_variables_collect = self.objective_function_class.design_variable.reverse_si_design_variables_collect(current_thermal_design_variables_collect)
            else:
                # generate the initial design variables collection by random values
                self.objective_function_class.design_variable.generate_si_design_variable_collect(self.individual_num, fixed_dict)

                # preprocess
                self.objective_function_class.design_variable.generate_therm_design_variable_collect()
                current_thermal_design_variables_collect = self.objective_function_class.design_variable.therm_design_variable_collect
                after_preprocess_design_variables_collect = self.preprocess_env_class.select_better_individuals(current_thermal_design_variables_collect)

                # convert the thermal design variables collection into the swarm intelligence design variables formation
                current_design_variables_collect = self.objective_function_class.design_variable.reverse_si_design_variables_collect(after_preprocess_design_variables_collect)
                current_thermal_design_variables_collect = after_preprocess_design_variables_collect

            print('Design Variables Number:', len(current_design_variables_collect))
            time.sleep(10)
        else:
            current_design_variables_collect = self.create_init_population()
            current_thermal_design_variables_collect = None

        start_time = time.time()
        # start learning
        for epoch in range(self.epochs):

            # print('current:', current_design_variables_collect)
            self.control_generation_num(epoch)
            # self.control_survival_coef(epoch)
            # set the populations (This process includes the meeting constraints)
            current_population = self.set_population(current_design_variables_collect, current_thermal_design_variables_collect)

            # save initial population
            # if epoch == 0 and initial_explore_type is False:
            #     file_number = 3  # ToDO if you change initial explore type, you will have to change another number
            #    self.save_initial_collect(current_population, file_number)

            # non dominated sort
            all_fronts, pareto_front = self.non_dominated_sort(current_population)

            # result class
            # save the pareto front and all design variables collection
            if result_dirs is not None:
                self.save_individual_dvs(all_fronts, pareto_front, result_dirs, fixed_dict, epoch)

            # draw objectives (for test)
            if epoch == self.epochs - 1:

                self.draw_population_place(all_fronts, pareto_front, epoch)

            # crowding sort
            next_design_variables_collect = self.crowding_sort(all_fronts)

            # crossover and mutation
            self.control_mutation_rate(epoch)
            next_design_variables_collect = self.evolutionary_process(next_design_variables_collect)
            # ToDo: add sample point
            next_design_variables_collect = self.add_sample(next_design_variables_collect)

            # update the design variables collection
            current_design_variables_collect = next_design_variables_collect

            # update the collection of thermal design variables
            if self.objective_function_class.name == 'Swarm':
                self.objective_function_class.design_variable.si_design_variable_collect = current_design_variables_collect
                self.objective_function_class.design_variable.generate_therm_design_variable_collect()
                current_thermal_design_variables_collect = self.objective_function_class.design_variable.therm_design_variable_collect

                self.objective_function_class.design_count = 0

        finish_time = time.time()
        print('Computational Time:', finish_time - start_time, '[s]')

        return next_design_variables_collect

    def explore_onetime(self, fixed_dict=None, result_dirs=None, initial_explore_type=False):
        # ToDo we have to make new afterprocessing_for_ga file
        """

        :param fixed_dict: type of dictionary which has the combination of names of fixed design variable and
                           values of it

        :param result_dirs: the list of directory names for restoring the computational results

        :param initial_explore_type: flag of determining better choice in two ways: creating initial population, load initial population

        :return: None

        Attributes
        -------------

        fixed_dict: dict

        result_dirs: list

        initial_explore_type: boolean


        Note:

            Algorithm:

            Part 1 : To reproduce prominent individuals

               0. loop 1-6 at epochs times

               1. create initial population

               2. calculate the objectives on the constraint condition + restore those values in the individual class

               3. operate non dominated sort and crowding sort

               4. operate evolutionary process(crossover, mutation)

               5. calculate the objectives on the constraint condition + restore those values in the individual class

               6. restore the data of individual class
                  => all_individual:list

            Part 2 : To select better individuals

               1. concatenate the data of individual class => flatten all_individual

               2. set decline step and mutation coefficient

               3. loop 3.1-3.3 at gene_step times

                  3.1. operate non dominated sort and crowding sort

                  3.2. conduct elite sort only

                  3.3. restore the each result in the result directory

        """

        # Part 1
        if initial_explore_type:

            all_elite_population = self.create_initial_elite_population()

        else:

            all_elite_population = self.collect_elite_individuals(fixed_dict)

            self.save_elite_individuals(all_elite_population)

        # Part 2
        last_population = self.extract_pareto_solution(all_elite_population, result_dirs, fixed_dict)

    def create_initial_elite_population(self):
        """
        create better initial population for onetime evolutionary strategy

        :return: all_elite_population
        """

        elite_population_path = './GA_results/EliteIndividual/{0}/'.format(
            self.objective_function_class.constraint_type)
        file_number = 2
        file_name = 'elite{}'.format(file_number)
        elite_population_path = elite_population_path + file_name
        design_variable_path = elite_population_path + '/design_variables_.npy'
        optimal_objective_path = elite_population_path + '/optimal_objectives_.npy'
        other_objective_path = elite_population_path + '/other_objectives_.npy'

        design_variable_arr = np.load(design_variable_path).tolist()
        optimal_objective_arr = np.load(optimal_objective_path).tolist()
        other_objective_arr = np.load(other_objective_path).tolist()

        all_elite_population = []

        for therm_design_variable, optimal_objective, other_objective in zip(design_variable_arr, optimal_objective_arr,
                                                                             other_objective_arr):
            # print(therm_design_variable, optimal_objective, other_objective)

            individual = Individual()
            individual.objectives = optimal_objective
            individual.therm_features = therm_design_variable
            # make normalized design variables
            norm_design_variable = self.objective_function_class.design_variable.reverse_si_design_variables_collect([therm_design_variable])
            individual.features = norm_design_variable[0]
            # run meet constraint
            meet_flag, other_objective_values = self.problem_class.calculate_objectives(individual)
            individual.other_objective_values = other_objective_values
            individual.dominates = functools.partial(self.problem_class.dominate, individual1=individual)

            if meet_flag:
                all_elite_population.append(individual)

        return all_elite_population

    def collect_elite_individuals(self, fixed_dict):
        """
        get the individuals at onetime elite strategy together

        :param fixed_dict:
        :return: all_elite_population
        """
        all_elite_population = []
        # comment out against print function is to cope with any errors or any bugs
        for epoch in range(self.epochs):
            # step 1
            # generate the initial design variables collection by random values
            self.objective_function_class.design_variable.generate_si_design_variable_collect(self.individual_num, fixed_dict)

            # print('Initial')
            # print(len(self.objective_function_class.design_variable.si_design_variable_collect))
            # time.sleep(5)

            # preprocess
            self.objective_function_class.design_variable.generate_therm_design_variable_collect()
            current_thermal_design_variables_collect = self.objective_function_class.design_variable.therm_design_variable_collect
            # print(len(self.objective_function_class.design_variable.therm_design_variable_collect))
            after_preprocess_design_variables_collect = self.preprocess_env_class.select_better_individuals(current_thermal_design_variables_collect)

            # print('after')
            # print(len(after_preprocess_design_variables_collect))
            # time.sleep(5)

            # convert the thermal design variables collection into the swarm intelligence design variables formation
            current_design_variables_collect = self.objective_function_class.design_variable.reverse_si_design_variables_collect(after_preprocess_design_variables_collect)
            current_thermal_design_variables_collect = after_preprocess_design_variables_collect

            print('Design Variables Number:', len(current_design_variables_collect))
            time.sleep(10)

            # step 2
            # meet constraint condition
            current_population = self.set_population(current_design_variables_collect, current_thermal_design_variables_collect)

            # step 3
            # non dominated sort
            all_fronts, pareto_front = self.non_dominated_sort(current_population)

            # crowding sort
            self.control_survival_coef(epoch)
            next_design_variables_collect = self.crowding_sort(all_fronts)

            # step 4
            # crossover and mutation
            self.control_mutation_rate(epoch)
            next_design_variables_collect = self.evolutionary_process(next_design_variables_collect)

            self.objective_function_class.design_variable.si_design_variable_collect = next_design_variables_collect
            self.objective_function_class.design_variable.generate_therm_design_variable_collect()
            next_thermal_design_variables_collect = self.objective_function_class.design_variable.therm_design_variable_collect

            # step 5
            next_population = self.set_population(next_design_variables_collect, next_thermal_design_variables_collect)

            # print('survival', len(next_population))
            # time.sleep(5)

            # step 6
            all_elite_population.extend(next_population)

            print('all', len(all_elite_population))
            time.sleep(10)

        return all_elite_population

    def extract_pareto_solution(self, all_elite_population, result_dirs, fixed_dict):
        """

        extract the better solution which meet optimal requirements from all elite population by elite evolutonary strategy

        :param all_elite_population: collection of individual class which went through elite sort
        :param result_dirs: the list which has both entire and pareto directory paths
        :param fixed_dict: {design variables name: constant value}
        :return: population
        """
        # Initialize generation number
        self.generation_num = len(all_elite_population)
        population = all_elite_population

        for gene_epoch in range(self.gene_step):
            # step 2
            self.control_generation_num(gene_epoch)

            # description
            print('')
            print('current generation epoch:', gene_epoch, 'current generation number:', self.generation_num)
            print('')

            # step 1
            all_fronts, pareto_fronts = self.non_dominated_sort(population)

            # draw pictures
            if gene_epoch % 2 == 0:
                self.draw_population_place(all_fronts, pareto_fronts, gene_epoch)

            # step 2
            self.save_individual_dvs(all_fronts, pareto_fronts, result_dirs, fixed_dict, gene_epoch)
            # step 3 elite sort
            # calculate objectives for sort
            objectives = self.select_elite(population)
            # extract sorting indexes
            objectives = list(np.argsort(objectives))
            population = [population[idx] for idx in objectives]
            # select the prior individuals
            population = population[:self.generation_num]

        return population

    # crossover
    def crossover(self, parent):
        """
        crossover, in particular two point crossover

        :param parent: two individuals for reproducing
        :return: children

        Attributes
        -------------------

        parent: list

        children: list

        """
        # set the parent design variables
        parent1, parent2 = parent[0], parent[1]
        # two point crossover
        idx = np.random.randint(0, self.design_variables_num - 1)
        jdx = np.random.randint(idx + 1, self.design_variables_num)

        # create child individuals
        # Initialize child individual by parent
        child1 = parent1.copy()
        child2 = parent2.copy()

        child1 = child1[:idx] + child2[idx:jdx] + child1[jdx:]
        child2 = child2[:idx] + child1[idx:jdx] + child2[jdx:]

        children = [child1, child2]

        return children

    # mutation
    def mutate(self, design_variables_collect):
        """
        mutation

        :param design_variables_collect: the collections of design variables
        :return: design_variables_collect

        Attributes
        ----------------

        design_variables_collect: list => [[x1, x2,...],[y1, y2,...], ...]
        """
        # the number of current design variables collection
        collection_num = len(design_variables_collect)

        if np.random.rand() < self.individual_mutate_rate:

            mutate_indiv_idx = np.random.randint(0, collection_num)

            if np.random.rand() < self.design_variable_mutate_rate:

                mutate_dv_idx = np.random.randint(0, self.design_variables_num)
                # put into random value
                design_variables_collect[mutate_indiv_idx][mutate_dv_idx] = np.random.rand()

        return design_variables_collect

    # evolutionary process (mutation + crossover)
    def evolutionary_process(self, design_variables_collect):
        """

        conduct the process of evolution such as crossover, mutation

        :param design_variables_collect: the collections of design variables
        :return: design_variables_collect

        Attributes
        ----------------

        design_variables_collect: list => [[x1, x2,...],[y1, y2,...], ...]
        """
        R = design_variables_collect.copy()
        Q_cross = design_variables_collect.copy()

        # print('the number of design variables collection:', len(Q_cross))

        # crossover process
        for _ in range(self.crossover_times * 4):
            # get the individual indexes which cross
            cross_index_1 = np.random.randint(0, len(Q_cross) - 1)
            cross_index_2 = np.random.randint(cross_index_1 + 1, len(Q_cross))

            # set the parent individuals
            parent = [Q_cross[cross_index_1], Q_cross[cross_index_2]]
            # crossover
            children = self.crossover(parent)

            for child in children:
                if child not in R:
                    R.append(child)

        # mutation process
        R = self.mutate(R)

        return R

    def select_elite(self, population):
        """

        create the standard of the elite sort

        :param population: the collections of individual class
        :return: objectives

        Attributes
        ------------------

        population: list => [individual class object1, individual class object2, ...]

        objectives: list
                    the collection of the values of euclid distance
        """

        objectives = []
        if self.elite_sort_objective_names == 'fuelburn':
            objectives = [individual.objectives[0] for individual in population]
        elif self.elite_sort_objective_names == 'euclid distance':
            objective_distances = [individual.objectives for individual in population]
            objectives = [np.sqrt(np.sum([obj ** 2 for obj in objectives])) for objectives in objective_distances]
        elif self.elite_sort_objective_names == 'normalized euclid distance':
            # objective_distances = [individual.normalized_objectives for individual in population]
            objectives = [individual.objectives for individual in population]
            objectives_arr = np.array(objectives)
            # make normalized objective's list
            objective_distances = []
            for idx in range(objectives_arr.shape[1]):
                target_arr = objectives_arr[:, idx]
                max_num, min_num = np.max(target_arr), np.min(target_arr)
                norm_objectives = []
                for obj in target_arr:
                    norm_obj = (obj - min_num) / (max_num - min_num)
                    norm_objectives.append(norm_obj)
                objective_distances.append(norm_objectives)

            # transpose
            objective_distances = np.array(objective_distances).T.tolist()
            objectives = [np.sqrt(np.sum([obj ** 2 for obj in objectives])) for objectives in objective_distances]

        return objectives

    # make the list of individual class
    # the list of individual class is population
    def set_population(self, design_variables_collect, thermal_design_variables_collect=None):
        """

        check whether the computational results which is available on the constraint condition or not and conduct elite sort

        :param design_variables_collect:
        :param thermal_design_variables_collect:
        :return: population
        """
        population = []

        if thermal_design_variables_collect is None:
            thermal_design_variables_collect = [None for _ in range(len(design_variables_collect))]

        for design_variables, therm_design_variables in zip(design_variables_collect, thermal_design_variables_collect):
            individual = Individual()
            individual.features = design_variables
            individual.therm_features = therm_design_variables
            # run meet constraint
            meet_flag, other_objective_values = self.problem_class.calculate_objectives(individual)
            individual.other_objective_values = other_objective_values
            individual.dominates = functools.partial(self.problem_class.dominate, individual1=individual)  # creative point for comparison of all design variables collection
            if meet_flag:
                population.append(individual)

        # operate elite sort according to pre-determined objective indexes
        objectives = self.select_elite(population)
        # extract sorting indexes
        objectives = list(np.argsort(objectives))
        population = [population[idx] for idx in objectives]
        # select the prior individuals
        population = population[:self.generation_num]
        return population

    def non_dominated_sort(self, population):
        """
        conduct non-dominated sort

        :param population: list of individual class

        :return: all_fronts, pareto_fronts
        """
        # list which puts into individual class object
        pareto_front = []  # pareto solutions
        all_fronts = []  # all solutions
        all_fronts.append([])

        for individual in population:
            individual.dominated_count = 0  # the number of dominated point
            individual.dominated_solutions = set()

            for other_individual in population:
                if individual.dominates(other_individual):
                    individual.dominated_solutions.add(other_individual)
                elif other_individual.dominates(individual):
                    individual.dominated_count += 1

            if individual.dominated_count == 0:
                individual.rank = 0
                pareto_front.append(individual)
                all_fronts[0].append(individual)

        idx = 0

        while len(all_fronts[idx]) > 0:
            rank_fronts = []

            for individual in all_fronts[idx]:
                for other_individual in individual.dominated_solutions:
                    # diminish the number of dominated points
                    other_individual.dominated_count -= 1

                    if other_individual.dominated_count == 0:
                        other_individual.rank = idx + 1
                        rank_fronts.append(other_individual)

            all_fronts.append(rank_fronts)
            idx += 1

        return all_fronts, pareto_front

    # helper function for crowding sort
    def calc_crowding_distance(self, rank_front):
        """
        calculate crowding distance by objective values

        :param rank_front: collection of individual class according to current rank

        :return: None
        """

        if len(rank_front) > 0:
            # Initialize crowding distance
            for individual in rank_front:
                individual.crowding_distance = 0
                self.problem_class.calculate_objectives(individual)

            # calculate crowding distance
            # print(rank_front[0].normalized_objectives)
            for m in range(len(rank_front[0].normalized_objectives)):
                # Initialize the edge distance
                front = sorted(rank_front, key=lambda x: x.normalized_objectives[m])
                front[0].crowding_distance = np.inf
                front[-1].crowding_distance = np.inf

                for idx, val in enumerate(front[1:-1]):
                    idx += 1
                    front[idx].crowding_distance += (front[idx + 1].normalized_objectives[m] - front[idx - 1].normalized_objectives[m])

    def crowding_sort(self, all_fronts):
        """
        give rank to each individual according to crowding distance and add the survival individual into next generation's individual

        :param all_fronts: collection of all individual class
        :return: next_design_variables
        """

        all_front_num = 0

        for front in all_fronts:
            for indiv in front:
                all_front_num += 1

        # if you want to extract better solutions, you have to change the ratio of survival according to epoch.
        survival_num = int(all_front_num * self.survival_coef)

        current_indiv_num = 0
        current_rank_number = 0

        for rank_front in all_fronts:

            current_indiv_num += len(rank_front)

            if current_indiv_num >= survival_num:
                break

            current_rank_number += 1

        diff_individual_num = survival_num - current_indiv_num

        next_design_variables_collect = []

        if diff_individual_num == 0:

            for rank in range(current_rank_number + 1):

                for individual in all_fronts[rank]:

                    next_design_variables_collect.append(individual.features)

        else:
            current_rank_number -= 1

            print('Crowding Sort Results')
            print('current rank number:', current_rank_number)
            print('current individual num:', current_indiv_num)
            print('')
            print('the number of fronts:', len(all_fronts))

            # if the rank number is first, we have to fix the rank number
            if current_rank_number <= 0:
                current_rank_number = 0

            additional_indiv_num = survival_num - (current_indiv_num - len(all_fronts[current_rank_number]))

            for rank in range(current_rank_number):

                for individual in all_fronts[rank]:

                    next_design_variables_collect.append(individual.features)

            current_rank_front = all_fronts[current_rank_number]

            # test error
            test_count = 0

            while len(next_design_variables_collect) <= survival_num:
                # calculate crowding distance
                self.calc_crowding_distance(current_rank_front)
                # make crowding distance list
                crowding_distances = [current_rank_front[idx].crowding_distance for idx in range(len(current_rank_front))]
                # change the order of individuals in the rank
                crowd_sort_idx = list(np.argsort(crowding_distances))
                ordered_rank_features = [current_rank_front[idx].features for idx in crowd_sort_idx]

                for design_variable in ordered_rank_features:

                    next_design_variables_collect.append(design_variable)

                test_count += 1

                if test_count == 1000:
                    break

        return next_design_variables_collect

    def save_elite_individuals(self, all_elite_population, result_path='./GA_results/EliteIndividual/'):
        """

        :param all_elite_population:
        :param result_path:
        :return: None
        """
        init_elite_path = result_path + self.objective_function_class.constraint_type
        filenames = os.listdir(init_elite_path)
        file_numbers = [f.split('e')[-1] for f in filenames]
        file_number = int(max(file_numbers)) + 1
        file_name = 'elite{}'.format(file_number)
        init_elite_path = init_elite_path + '/' + file_name
        os.mkdir(init_elite_path)

        design_variables_arr = []
        optimal_objective_arr = []
        other_objective_arr = []

        for individual in all_elite_population:
            design_variables_arr.append(individual.therm_features)
            optimal_objective_arr.append(individual.objectives)
            other_objective_arr.append(individual.other_objective_values)

        design_variables_arr = np.array(design_variables_arr)
        optimal_objective_arr = np.array(optimal_objective_arr)
        other_objective_arr = np.array(other_objective_arr)

        design_variables_path = init_elite_path + '/design_variables_.npy'
        optimal_objective_path = init_elite_path + '/optimal_objectives_.npy'
        other_objective_path = init_elite_path + '/other_objectives_.npy'

        # save
        np.save(design_variables_path, design_variables_arr)
        np.save(optimal_objective_path, optimal_objective_arr)
        np.save(other_objective_path, other_objective_arr)

    def save_individual_dvs(self, all_fronts, pareto_fronts, result_dirs, fixed_dict, epoch):

        """
        restore the data of design variables in the predetermined directry

        :param all_fronts: the collection of all individual class objects
        :param pareto_fronts: the collection of pareto individual class objects
        :param result_dirs: the paths of memorizing the data of both fronts
        :param fixed_dict: dictionary which contains the constant value and its name
        :param epoch: current epoch
        :return: None
        """
        # set the directory path for saving the design variables
        pareto_result_dir, all_result_dir = result_dirs

        pareto_result_dir += '/epoch{}'.format(epoch)
        all_result_dir += '/epoch{}'.format(epoch)

        if not os.path.exists(pareto_result_dir):
            os.mkdir(pareto_result_dir)

        if not os.path.exists(all_result_dir):
            os.mkdir(all_result_dir)

        # make filename for both pareto and entire results
        fix_index_names = ''
        if fixed_dict is not None:
            for key, val in fixed_dict.items():
                fix_index_names += '{}{}'.format(key, str(val))

        # extract the array of design variables from the individual class
        pareto_design_variables = []
        pareto_optimal_objectives = []
        pareto_other_objectives = []

        for pareto_individual in pareto_fronts:
            pareto_design_variable = pareto_individual.therm_features
            pareto_optimal_objective = pareto_individual.objectives
            pareto_other_objective = pareto_individual.other_objective_values

            pareto_design_variables.append(pareto_design_variable)
            pareto_optimal_objectives.append(pareto_optimal_objective)
            pareto_other_objectives.append(pareto_other_objective)

        all_design_variables = []
        all_optimal_objectives = []
        all_other_objectives = []

        for each_front in all_fronts:
            for all_individual in each_front:
                all_design_variable = all_individual.therm_features
                all_optimal_objective = all_individual.objectives
                all_other_objective = all_individual.other_objective_values

                all_design_variables.append(all_design_variable)
                all_optimal_objectives.append(all_optimal_objective)
                all_other_objectives.append(all_other_objective)

        # change the numpy array
        pareto_design_variables = np.array(pareto_design_variables)
        all_design_variables = np.array(all_design_variables)
        pareto_optimal_objectives = np.array(pareto_optimal_objectives)
        all_optimal_objectives = np.array(all_optimal_objectives)
        pareto_other_objectives = np.array(pareto_other_objectives)
        all_other_objectives = np.array(all_other_objectives)

        # save as npy file
        target_filename = '/{}_{}'.format('design_variables', fix_index_names)
        pareto_filename = pareto_result_dir + target_filename
        all_filename = all_result_dir + target_filename
        np.save(pareto_filename, pareto_design_variables)
        np.save(all_filename, all_design_variables)

        target_filename = '/{}_{}'.format('optimal_objectives', fix_index_names)
        pareto_filename = pareto_result_dir + target_filename
        all_filename = all_result_dir + target_filename
        np.save(pareto_filename, pareto_optimal_objectives)
        np.save(all_filename, all_optimal_objectives)

        target_filename = '/{}_{}'.format('other_objectives', fix_index_names)
        pareto_filename = pareto_result_dir + target_filename
        all_filename = all_result_dir + target_filename
        np.save(pareto_filename, pareto_other_objectives)
        np.save(all_filename, all_other_objectives)

    def draw_population_place(self, all_fronts, pareto_fronts, epoch):
        """

        :param all_fronts:
        :param pareto_fronts:
        :param epoch:
        :return: None
        """
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt

        all_objective_data = []
        for front in all_fronts:
            for all_individual in front:
                if self.optimal_dir == 'upperright':
                    obj = -1 * np.array(all_individual.objectives)
                    obj = obj.tolist()
                else:
                    obj = all_individual.objectives

                all_objective_data.append(obj)

        all_objective_data = np.array(all_objective_data)

        pareto_front_objective_data = []
        for individual in pareto_fronts:
            if self.optimal_dir == 'upperright':
                obj = -1 * np.array(individual.objectives)
                obj = obj.tolist()
            else:
                obj = individual.objectives
            pareto_front_objective_data.append(obj)

        pareto_front_objective_data = np.array(pareto_front_objective_data)

        if self.objective_nums == 3:
            fig = plt.figure()
            ax = Axes3D(fig)

            ax.scatter(all_objective_data[:, 0], all_objective_data[:, 1], all_objective_data[:, 2], c='b',
                       alpha=0.2)

            ax.scatter(pareto_front_objective_data[:, 0], pareto_front_objective_data[:, 1],
                       pareto_front_objective_data[:, 2], c='r')

            # ax.set_xlim([max(all_objective_data[:, 0]), min(all_objective_data[:, 0])])
            ax.set_ylim([max(all_objective_data[:, 1]), min(all_objective_data[:, 1])])
            # ax.set_zlim([max(all_objective_data[:, 2]), min(all_objective_data[:, 2])])

            plt.show()

        elif self.objective_nums == 2:

            plt.figure()
            plt.scatter(all_objective_data[:, 1], all_objective_data[:, 0], c='b', alpha=0.1)
            plt.scatter(pareto_front_objective_data[:, 1], pareto_front_objective_data[:, 0], c='r')
            plt.xlabel('engine weight[kg]')
            plt.ylabel('fuelburn[kg]')
            plt.title('Two Multi Objective Optimization epoch {}'.format(epoch))
            plt.show()


# non dominated sort
# subtract the pareto front from the design variables collect
# save the pareto front individual and current individuals
# evolution or mutate process
# go to next
# Above all, implement the all process and function in multi_ga class
# update design variables collection


def test_mo():
    # objective function name list
    objective_indexes = ['f1', 'f2']
    # the number of objective values
    objective_nums = len(objective_indexes)
    # bounds
    bounds = [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]]
    # optimization direction
    optimal_dir = 'upperright'
    # the number of variables
    design_variable_nums = 6
    # build the objective function class
    objective_function_class = ObjectTestFunc(objective_nums, design_variable_nums, bounds)

    individual_num = 500
    crossover_times = 200
    epochs = 12

    nsga2 = NSGA2(objective_indexes, objective_function_class, individual_num, epochs, optimal_dir=optimal_dir, crossover_times=crossover_times)

    nsga2.explore()


def test_ga():
    objective_indexes = {'fuel_weight': 0, 'engine_weight': 1}

    # make integration env swarm class arguments
    # BaseLine Arguments
    baseline_aircraft_name = 'A320'
    baseline_aircraft_type = 'normal'
    baseline_engine_name = 'V2500'
    baseline_propulsion_type = 'turbofan'

    # data path
    aircraft_data_path = './DataBase/aircraft_test.json'
    engine_data_path = './DataBase/engine_test.json'
    mission_data_path = './Missions/fuelburn16000.json'
    # mission_data_path = './Missions/cargo0.8_passenger1.0_.json'

    # create database arguments
    data_base_args = [aircraft_data_path, engine_data_path, mission_data_path]

    baseline_args = [(baseline_aircraft_name, baseline_aircraft_type), (baseline_engine_name, baseline_propulsion_type),
                     (aircraft_data_path, engine_data_path)]

    # off design parameters
    off_altitude = 0.0
    off_mach = 0.0
    off_required_thrust = 133000  # [N]

    off_param_args = [off_altitude, off_mach, off_required_thrust]

    # current args
    current_aircraft_name = 'A320'
    current_aircraft_type = 'normal'
    current_engine_name = 'V2500'
    current_propulsion_type = 'turbofan'

    current_args = [(current_aircraft_name, current_aircraft_type), (current_engine_name, current_propulsion_type),
                    (aircraft_data_path, engine_data_path)]

    # Build Overall Exploration class
    # constraint type
    constraint_type = 'TIT'

    design_point = 'cruise'

    # env_swarm_args = [baseline_args, current_args, off_param_args, mission_data_path, constraint_type]
    # build swarm intelligent class
    ies = IntegrationEnvSwarm(baseline_args, current_args, off_param_args, mission_data_path, constraint_type, design_point)

    # range tuning
    v2500_design_variables = [4.7, 30.0, 1.66, 1380]
    ies.range_tuning(v2500_design_variables)

    # replace the class
    objective_function_class = ies

    # optimization direction
    optimal_dir = 'downleft'

    # GeneticAlgorithm global variables
    individual_num = 300
    crossover_times = 200
    epochs = 2

    # build the preprocess class for exploration class
    preprocess_env_class = PreprocessIntegrationEnv(current_aircraft_name, current_engine_name, current_aircraft_type,
                                                    current_propulsion_type, data_base_args)

    # build the multi-objective optimization class
    # if you want to get the better results, you have to implement the preprocess function like surviving the better
    # individuals
    nsga2 = NSGA2(objective_indexes, objective_function_class, individual_num, epochs, optimal_dir, preprocess_env_class, crossover_times)

    # fixed dict
    fixed_dict = {'OPR': 30}

    # run the multi-objective function
    nsga2.explore(fixed_dict)


if __name__ == '__main__':
    test_mo()
    # test_ga()
