import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


class ResultIndividual(object):

    def __init__(self):
        self.thermal_design_variables = None
        self.optimal_objectives = None
        self.objectives = None

        # in case of adding the objective index, you have to change following part
        # the name list of objective indexes
        self.names_of_objective_indexes = {'fuel_weight': 0, 'engine_weight': 1, 'aircraft_weight': 2,
                                           'max_takeoff_weight': 3, 'electric_weight': 4,
                                           'isp': 5, 'sfc': 6, 'co_blade_h': 7, 'th_ew_ratio': 8}
        # objective indexes (default)
        self.fuel_weight = None
        self.engine_weight = None
        self.aircraft_weight = None
        self.max_takeoff_weight = None
        self.electric_weight = None  # the weight of electric devices
        self.isp = None
        self.sfc = None
        self.co_blade_h = None  # the height of final stage of compressor
        self.stage_numbers = None  # the list of compressor stage numbers
        self.th_ew_ratio = None  # the ratio of thrust at ground and engine weight

    def set_value_from_results(self, result_arr):
        optimal_objective_arr, thermal_design_variable_arr, other_objective_arr = result_arr
        # create objective arr
        objective_arr = optimal_objective_arr + other_objective_arr
        # print(objective_arr)

        self.fuel_weight = objective_arr[self.names_of_objective_indexes['fuel_weight']]
        self.engine_weight = objective_arr[self.names_of_objective_indexes['engine_weight']]
        self.aircraft_weight = objective_arr[self.names_of_objective_indexes['aircraft_weight']]
        self.max_takeoff_weight = objective_arr[self.names_of_objective_indexes['max_takeoff_weight']]
        self.electric_weight = objective_arr[self.names_of_objective_indexes['electric_weight']]
        self.isp = objective_arr[self.names_of_objective_indexes['isp']]
        self.sfc = objective_arr[self.names_of_objective_indexes['sfc']]
        self.co_blade_h = objective_arr[self.names_of_objective_indexes['co_blade_h']]
        self.th_ew_ratio = objective_arr[self.names_of_objective_indexes['th_ew_ratio']]

        self.thermal_design_variables = thermal_design_variable_arr
        self.optimal_objectives = optimal_objective_arr
        self.objectives = objective_arr


# utils for afterprocess

def generate_result_pathes(date):
    # Initialize base directory name
    result_dir_name = date
    # Initialize necessary variables
    pareto_result_dir = None
    all_result_dir = None
    double_count = 0
    for _ in range(10):
        root_dir = os.listdir(result_dir_name)
        # print(root_dir)
        root_dir = [root for root in root_dir if root != 'memo.txt']
        if len(root_dir) == 0:
            break
        # you should take choice to the number of learning steps except for 2
        elif len(root_dir) == 2:
            pareto_result_dir = result_dir_name + '/' + root_dir[0]
            all_result_dir = result_dir_name + '/' + root_dir[1]
            break
        else:
            result_dir_name += '/' + root_dir[0]

    # print('Base File Name')
    # print(pareto_result_dir)
    # print(all_result_dir)
    pareto_filenames = os.listdir(pareto_result_dir)
    all_filenames = os.listdir(all_result_dir)

    # the length of the variable called all_filenames or pareto_filenames is equal to the learning times
    pareto_filenames = [os.listdir(pareto_result_dir + '/' + file) for file in pareto_filenames]
    pareto_filename_pathes = []
    for epoch, pareto_file_path in enumerate(pareto_filenames):
        epoch_file_path = []
        for file in pareto_file_path:
            file_path = pareto_result_dir + '/epoch{}/'.format(epoch) + file
            epoch_file_path.append(file_path)

        pareto_filename_pathes.append(epoch_file_path)

    all_filenames = [os.listdir(all_result_dir + '/' + file) for file in all_filenames]
    all_filename_pathes = []
    for epoch, files in enumerate(all_filenames):
        epoch_file_path = []
        for file in files:
            file_path = all_result_dir + '/epoch{}'.format(epoch) + '/' + file
            epoch_file_path.append(file_path)
        all_filename_pathes.append(epoch_file_path)

    # confirmation
    # print(pareto_filename_pathes)
    # print('')
    # print(all_filename_pathes)

    return pareto_filename_pathes, all_filename_pathes


def create_individual_class_collect(epoch, filenames):
    each_individuals = []
    target_epoch_filename_pathes = filenames[epoch]

    # print(target_epoch_filename_pathes)

    optimal_objective_path, other_objective_path, design_variable_path = target_epoch_filename_pathes
    optimal_objective_arr = np.load(optimal_objective_path)
    design_variable_arr = np.load(design_variable_path)
    other_objective_arr = np.load(other_objective_path)

    result_arrs = [optimal_objective_arr.tolist(), design_variable_arr.tolist(), other_objective_arr.tolist()]
    result_arrs = np.array(result_arrs)

    for idx in range(result_arrs.shape[1]):
        result_arr = list(result_arrs[:, idx])
        individual = ResultIndividual()
        individual.set_value_from_results(result_arr)
        each_individuals.append(individual)

    return each_individuals


# Describe data in the individual class
def describe_individual_class_data(individuals):

    for number, individual in enumerate(individuals):
        print('')
        print('-' * 100)
        print('Design Number: {}'.format(number + 1))
        print('Thermal Design Variables:', individual.thermal_design_variables)
        print('FuelBurn [kg]:', individual.fuel_weight)
        print('Engine Weight [kg]:', individual.engine_weight)
        print('Aircraft Weight [kg]:', individual.aircraft_weight)
        print('Max Takeoff Weight [kg]:', individual.max_takeoff_weight)
        print('Electric Weight [kg]:', individual.electric_weight)
        print('Specific Thrust :', individual.isp)
        print('Specific Fuel Consumption:', individual.sfc)
        print('Height of Final Stage Compressor [m]:', individual.co_blade_h)
        print('Stage number')
        print('Ratio of thrust at ground and engine weight:', individual.th_ew_ratio)
        print('-' * 100)
        print('')


# create result individuals of every epoch
def create_total_individuals(filename_paths, epoch_list):
    pareto_filename_paths, all_filename_paths = filename_paths
    total_individuals = []

    for epoch in epoch_list:
        pr_individual = create_individual_class_collect(epoch=epoch, filenames=pareto_filename_paths)
        all_individual = create_individual_class_collect(epoch=epoch, filenames=all_filename_paths)
        target_individual = [all_individual, pr_individual]
        total_individuals.append(target_individual)

    return total_individuals


# create objective indexes list
def create_indexes(individuals):
    ew, fb, ele, air, mtow = [], [], [], [], []
    bpr, opr, fpr, tit, bpre, fpre = [], [], [], [], [], []
    sfc, isp = [], []
    cobh = []
    th_ew_ratio = []
    for individual in individuals:
        ew.append(individual.engine_weight)
        fb.append(individual.fuel_weight)
        ele.append(individual.electric_weight)
        air.append(individual.aircraft_weight)
        mtow.append(individual.max_takeoff_weight)

        dv = individual.thermal_design_variables
        bpr.append(dv[0])
        opr.append(dv[1])
        fpr.append(dv[2])
        tit.append(dv[3])
        bpre.append(dv[4])
        fpre.append(dv[5])

        sfc.append(individual.sfc)
        isp.append(individual.isp)
        cobh.append(individual.co_blade_h)
        th_ew_ratio.append(individual.th_ew_ratio)

    return [ew, fb, ele, air, mtow, bpr, opr, fpr, tit, bpre, fpre, sfc, isp, cobh, th_ew_ratio]


# draw pictures of indicating the relationship with each objective indexes
def draw_pictures(index_names, results, target_objectives, xlim, ylim, save=False, picture_dir=None, annotation=False):
    objective_index_dicts = {'Engine Weight': 0, 'Fuel Burn': 1, 'Electric Weight': 2, 'Aircraft Weight': 3,
                             'Max Takeoff Weight': 4, 'BPR': 5, 'OPR': 6, 'FPR': 7, 'TIT': 8, 'BPRe': 9, 'FPRe': 10,
                             'Specific Fuel Consumption': 11, 'Specific Thrust': 12, 'Height of Final Compressor': 13,
                             'Ratio of thrust at ground and engine weight': 14}

    reverse_objective_index_dicts = {}
    for key, val in objective_index_dicts.items():
        reverse_objective_index_dicts[val] = key

    markers = ['o', 's', 'p', '+', '<', '^', 'D']
    colors = ['r', 'g', 'b', 'c', 'm', 'k', 'w']
    index1, index2 = index_names
    index1, index2 = objective_index_dicts[index1], objective_index_dicts[index2]
    fig, ax = plt.subplots(figsize=(10, 6))

    for epoch, epoch_results in enumerate(results):
        all_results, pr_results = epoch_results
        size = 30
        pareto_alpha = 1.0
        all_alpha = 0.1
        marker = markers[epoch]
        color = colors[epoch]
        ax.scatter(all_results[index1], all_results[index2], s=size, c=color, marker=marker,
                    label='entire_epoch{}'.format(epoch), alpha=all_alpha)

    diff_x = xlim[1] - xlim[0]
    diff_y = ylim[1] - ylim[0]
    annot_coef = 0.15

    ax.scatter(pr_results[index1], pr_results[index2], s=60, c='r', marker=marker, label='pareto_epoch{}'.format(epoch), alpha=pareto_alpha)

    ax.scatter(target_objectives[0], target_objectives[1], s=200, marker='*')

    if annotation:
        ax.annotate(
            '{}:{}'.format(reverse_objective_index_dicts[index1], round(pr_results[index1][0], 2)) + ',' + '{}:{}'.format(
                reverse_objective_index_dicts[index2], round(pr_results[index2][0], 2)),
            (pr_results[index1][0], pr_results[index2][0]), size=10,
            xytext=(pr_results[index1][0] + diff_x * annot_coef, pr_results[index2][0] - diff_y * annot_coef),
            arrowprops=dict())

        """
        ax.annotate(
            '{}:{}'.format(reverse_objective_index_dicts[index1], round(pr_results[index1][-1], 2)) + ',' + '{}:{}'.format(
                reverse_objective_index_dicts[index2], round(pr_results[index2][-1], 2)),
            (pr_results[index1][-1], pr_results[index2][-1]), size=10,
            xytext=(pr_results[index1][-1] - diff_x * annot_coef, pr_results[index2][-1] + diff_y * annot_coef),
            arrowprops=dict())
        """


        ax.annotate('V2500  {}:{}, {}:{}'.format(reverse_objective_index_dicts[index1], target_objectives[0],
                                                 reverse_objective_index_dicts[index2], target_objectives[1]),
                    (target_objectives[0], target_objectives[1]), size=10,
                    xytext=(target_objectives[0], target_objectives[1] + diff_y * annot_coef * 1.1), arrowprops=dict())

    plt.xlabel(index_names[0])
    plt.ylabel(index_names[1])
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend(loc='best')
    plt.grid()
    if save:
        plt.savefig(picture_dir)
    plt.show()

def draw_pictures_for_comparison(index_names, comparison_results, target_objectives, xlim, ylim, save=False, picture_dir=None):
    objective_index_dicts = {'Engine Weight': 0, 'Fuel Burn': 1, 'Electric Weight': 2, 'Aircraft Weight': 3,
                             'Max Takeoff Weight': 4, 'BPR': 5, 'OPR': 6, 'FPR': 7, 'TIT': 8, 'BPRe': 9, 'FPRe': 10,
                             'Specific Fuel Consumption': 11, 'Specific Thrust': 12, 'Height of Final Compressor': 13,
                             'Ratio of thrust at ground and engine weight': 14}

    markers = ['o', 's', 'p', '+', '<', '^', 'D']
    colors = ['r', 'g', 'b', 'c', 'm', 'k']
    labels = ['TIT', 'COT', 'COBH', 'DFAN', 'LFAN']  # the list of names of constraint type. if you add another constraint, you will have to fix this part
    index1, index2 = index_names
    index1, index2 = objective_index_dicts[index1], objective_index_dicts[index2]
    plt.figure(figsize=(10, 6))

    for idx, results in enumerate(comparison_results):
        all_results, pr_results = results
        all_size = 20
        pareto_size = 50
        all_alpha = 0.1
        pareto_alpha = 1.0
        color = colors[idx]
        marker = markers[idx]
        label = labels[idx]

        plt.scatter(all_results[index1], all_results[index2], s=all_size, c=color, marker=marker, label=label, alpha=all_alpha)
        plt.scatter(pr_results[index1], pr_results[index2], s=pareto_size, c=color, marker=marker, label=label, alpha=pareto_alpha)

    plt.scatter(target_objectives[0], target_objectives[1], s=100, marker='*')

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(index_names[0])
    plt.ylabel(index_names[1])
    plt.legend()
    plt.grid()
    if save:
        plt.savefig(picture_dir + '/' + '{0}_{1}.png'.format(index_names[0], index_names[1]))
    plt.show()



def construct_results(total_individuals):
    results = []
    for individuals in total_individuals:
        all_individuals, pareto_individuals = individuals
        all_results = create_indexes(all_individuals)
        pr_results = create_indexes(pareto_individuals)
        results.append([all_results, pr_results])

    return results







