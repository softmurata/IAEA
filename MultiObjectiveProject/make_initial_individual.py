import numpy as np


def replace_initial_population_dir(element_path, save_dir, constraint_type, file_number):
    # set filename path
    save_filename = save_dir + '/{0}/{1}.npy'.format(constraint_type, file_number)
    # load data
    dvs = np.load(element_path)
    # save another file
    np.save(save_filename, dvs)


def combine_initial_population_dir(element_paths, save_dir, constraint_type, file_number):
    # set filename path
    save_filename = save_dir + '/{0}/{1}.npy'.format(constraint_type, file_number)

    new_dvs = []
    for element_path in element_paths:
        dvs = np.load(element_path).tolist()
        new_dvs.extend(dvs)

    new_dvs = np.array(new_dvs)

    np.save(save_filename, new_dvs)


if __name__ == '__main__':

    element_path = './GA_results/20180913-1/A320_normal_V2500_turbofan/fuel_weight_engine_weight_TIT1820/entire/epoch0/design_variables_.npy'
    # element_path = './GA_results/20180913-3/A320_normal_V2500_turbofan/fuel_weight_engine_weight_COT965/entire/epoch0/design_variables_.npy'
    save_dir = './GA_results/InitialIndividual'
    constraint_type = 'TIT'
    file_number = 1

    replace_initial_population_dir(element_path, save_dir, constraint_type, file_number)

    element_paths = ['./GA_results/20180913-1/A320_normal_V2500_turbofan/fuel_weight_engine_weight_TIT1820/entire/epoch0/design_variables_.npy']

    combine_initial_population_dir(element_paths, save_dir, constraint_type, file_number)






