import numpy as np
import json
import os
from multi_ga import NSGA2
from integration_env_si import IntegrationEnvSwarm
from preprocess_for_integenv import PreprocessIntegrationEnv
from constraints import EnvConstraint


# If you want to change the mission or types, you fix following class
class SetGA(object):
    """
    Attributes
    --------------
    aircraft_data_path: str

    engine_data_path: str

    mission_data_path: str

    baseline_aircraft_name: str

    baseline_aircraft_type: str

    baseline_engine_name: str

    baseline_propulsion_type: str

    current_aircraft_name: str

    current_aircraft_type: str

    current_engine_name: str

    current_propulsion_type: str

    off_altitude: float

    off_mach: float

    off_required_thrust: float

    objective1: str

    objective2: str

    objective3; str

    constraint type: str

    constraint value: float

    range tuning: boolean

    target design variables: list

    optimal_dir: str

    individual_num: int

    crossover_times: int

    epochs: int

    gene_step: int

    survival_coef: float

    decline_step: int

    elite_sort_objective_names: str

    """

    def __init__(self, mission_data_path='./Missions/cargo1.0_passenger1.0_.json'):

        self.aircraft_data_path = './DataBase/aircraft_test.json'
        self.engine_data_path = './DataBase/engine_test.json'
        self.mission_data_path = mission_data_path

        # Default value
        self.baseline_aircraft_name = None
        self.baseline_aircraft_type = None
        self.baseline_engine_name = None
        self.baseline_propulsion_type = None

        self.current_aircraft_name = None
        self.current_aircraft_type = None
        self.current_engine_name = None
        self.current_propulsion_type = None

        # Default set parameters
        self.off_altitude = None
        self.off_mach = None
        self.off_required_thrust = None
        self.objective1 = None
        self.objective2 = None
        self.objective3 = None
        self.constraint_type = None
        self.constraint_value = None
        self.design_point = None
        self.target_design_variables = None

        self.optimal_dir = None
        self.individual_num = None
        self.crossover_times = None
        self.epochs = None
        self.gene_step = None
        self.survival_coef = None
        self.decline_step = None
        self.elite_sort_objective_names = None
        self.strategy = None

    def set_data_base_args(self):
        """
        create the list of paths of data base
        """

        return [self.aircraft_data_path, self.engine_data_path, self.mission_data_path]

    def set_baseline_args(self):
        """
        set baseline argument's list

        List shape:  [(name, type),...]

        """
        # BaseLine Arguments
        self.baseline_aircraft_name = 'A320'
        self.baseline_aircraft_type = 'normal'
        self.baseline_engine_name = 'V2500'
        self.baseline_propulsion_type = 'turbofan'
        baseline_args = [(self.baseline_aircraft_name, self.baseline_aircraft_type),
                         (self.baseline_engine_name, self.baseline_propulsion_type),
                         (self.aircraft_data_path, self.engine_data_path)]
        return baseline_args

    def set_current_args(self):
        """
        set current argument's list
        """
        self.current_aircraft_name = 'A320'
        self.current_aircraft_type = 'normal'
        self.current_engine_name = 'V2500'
        self.current_propulsion_type = 'TeDP'  #'TeDP'

        current_args = [(self.current_aircraft_name, self.current_aircraft_type),
                        (self.current_engine_name, self.current_propulsion_type),
                        (self.aircraft_data_path, self.engine_data_path)]

        return current_args

    def set_off_param_args(self):
        """
        create the list of parameters ar off design point

        off_param_args = [off_altitude[m], off_mach, off_required_thrust[N]]
        """
        # off design parameters
        off_altitude = 0.0
        off_mach = 0.0
        off_required_thrust = 133000  # [N]

        self.design_point = ['cruise', [10668, 0.78]]

        off_param_args = [off_altitude, off_mach, off_required_thrust]
        self.off_altitude = off_altitude
        self.off_mach = off_mach
        self.off_required_thrust = off_required_thrust

        return off_param_args

    def set_objective_indexes(self):
        """
        set the name of objective index

        ex) 'fuel weight', 'engine weight', 'electric weight',...

        """

        objective1 = 'fuel_weight'
        # objective2 = 'electric_weight'  # 4
        objective2 = 'engine_weight'  # 1
        objective3 = 'takeoff_weight'  # 3

        self.objective1 = objective1
        self.objective2 = objective2
        self.objective3 = objective3

        return {objective1: 0, objective2: 1}

    def set_constraint_type(self):
        """
        set constraint type and constraint value

        ex) (TIT, 1820), (COT, 965), (COBH, 0.015), (DFAN, 2.0)

        """

        constraint_type = 'TIT'  # 'COT', 'compressor_out_blade_height', 'front_diameter', 'width_length_distributed_fan'
        constraint_class = EnvConstraint()
        constraint_value = constraint_class.get_constraint_target(constraint_type)
        tech_lev = 2.85
        self.constraint_type = constraint_type
        self.constraint_value = constraint_value
        self.tech_lev = tech_lev

        return constraint_type, constraint_value, tech_lev

    def set_range_tuning(self):
        """
        confirm the need of range tuning and set the baseline design variables

        :return: range_tuning, target_design_variables

        """
        engine_file = open(self.engine_data_path, 'r')
        engine_file = json.load(engine_file)
        target_engine_file = engine_file[self.baseline_engine_name]
        target_design_variables = target_engine_file['design_variable']
        self.target_design_variables = target_design_variables

        return target_design_variables

    def set_swarm_intelligence_args(self):
        """
        give the hyper parameters of swarm intelligence optimization

        :return: optimal_dir, individual_num, crossover_times, epochs, gene_step, survival_coef, decline_step, elite_sort_objective_names

        Attributes
        -----------
        optimal_dir: str
                     'downleft(, 'upperright'

        elite_sort_objective_names: str
                                    'fuelburn', 'euclid distance', 'normalized euclid distance'

        strategy: str
                  evolutionary strategy => 'normal' or 'onetime'
        """
        optimal_dir = 'downleft'  # otherwise 'upperright'
        individual_num = 5000
        crossover_times = 350
        epochs = 7
        gene_step = 5
        survival_coef = 0.5
        decline_step = 10
        elite_sort_objective_names = 'fuelburn'  # 'normalized euclid distance', 'fuelburn'
        strategy = 'normal'  # 'normal' => evolutionary strategy

        self.optimal_dir = optimal_dir
        self.individual_num = individual_num
        self.crossover_times = crossover_times
        self.epochs = epochs
        self.gene_step = gene_step
        self.survival_coef = survival_coef
        self.decline_step = decline_step
        self.elite_sort_objective_names = elite_sort_objective_names
        self.strategy = strategy

        return optimal_dir, individual_num, crossover_times, epochs, gene_step, survival_coef, decline_step, \
               elite_sort_objective_names

    def set_fixed_dict(self):
        """
        set fixed dict

        ex) {'OPR': 30, 'TIT': 1400}

        """

        # fixed_dict = {'FPR': 1.6}

        # BWB
        # {'xw': 1.0, 'u1': 0.8 , 'v1': 0.48, 'u2': 1.0, 'v2': 0.2}
        fixed_dict = None

        return fixed_dict

    def save_memory(self, memory_dir):
        """

        :param memory_dir: path of memory directory
        :return: None
        """
        # write something about investigation into text file
        memory_path = memory_dir + '/memo.txt'
        content_dictionary = {'current_aircraft_name': self.current_aircraft_name,
                              'current_aircraft_type': self.current_aircraft_type,
                              'current_propulsion_type': self.current_propulsion_type,
                              'current_engine_name': self.current_engine_name,
                              'off_altitude': self.off_altitude,
                              'off_mach': self.off_mach,
                              'off_required_thrust': self.off_required_thrust,
                              'objective1': self.objective1,
                              'objective2': self.objective2,
                              'constraint_type': self.constraint_type,
                              'constraint_value': self.constraint_value,
                              'tech level': self.tech_lev,
                              'optimal_dir': self.optimal_dir,
                              'individual_num': self.individual_num,
                              'crossover_times': self.crossover_times,
                              'epochs': self.epochs,
                              'gene_step': self.gene_step,
                              'survival_coefficient': self.survival_coef,
                              'decline_step': self.decline_step,
                              'elite_sort_objective_names': self.elite_sort_objective_names,
                              'strategy': self.strategy,
                              'design point': self.design_point}

        with open(memory_path, 'w') as f:

            for key, val in content_dictionary.items():
                f.write(key + ':' + str(val) + '\n')

            f.close()


def main_ga():
    # Input the date
    fuelburn = 11000
    date = 'tedpfb{}ele026epoch7'.format(fuelburn)

    # mission_data_path = './Missions/maxpayload_base.json'

    mission_data_path = './Missions/fuelburn{}.json'.format(fuelburn)

    # build class for setting arguments
    sga = SetGA(mission_data_path)

    baseline_args = sga.set_baseline_args()
    current_args = sga.set_current_args()
    data_base_args = sga.set_data_base_args()
    objective_indexes = sga.set_objective_indexes()
    off_param_args = sga.set_off_param_args()

    # Build Overall Exploration class
    # constraint type
    constraint_type, constraint_value, tech_level = sga.set_constraint_type()

    design_point = sga.design_point

    
    # build swarm intelligent class
    ies = IntegrationEnvSwarm(baseline_args, baseline_args, off_param_args, mission_data_path, constraint_type, design_point)


    
    # range tuning
    target_design_variables = sga.set_range_tuning()

    ies.range_tuning(target_design_variables)

    cruise_range = ies.cruise_range
    

    # build swarm intelligent class
    ies = IntegrationEnvSwarm(baseline_args, current_args, off_param_args, mission_data_path, constraint_type,
                              design_point)
    
    ies.set_range(cruise_range)

    ies.set_tech_lev(tech_level)
    

    # replace the class
    objective_function_class = ies

    optimal_dir, individual_num, crossover_times, epochs, gene_step, survival_coef, decline_step, elite_sort_objective_names = sga.set_swarm_intelligence_args()

    # build the preprocess class for exploration class
    preprocess_env_class = PreprocessIntegrationEnv(sga.current_aircraft_name, sga.current_engine_name, sga.current_aircraft_type,
                                                    sga.current_propulsion_type, data_base_args, design_point)

    # build after process data path
    result_dir = './GA_results/' + date

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # save memory text file
    memory_dir = result_dir
    sga.save_memory(memory_dir)

    result_dir += '/' + sga.current_aircraft_name + '_' + sga.current_aircraft_type + '_' + sga.current_engine_name + '_' + sga.current_propulsion_type

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    objective_index_names = ''

    for key in objective_indexes.keys():
        objective_index_names += key + '_'

    result_dir += '/' + objective_index_names + constraint_type + str(constraint_value)

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    pareto_result_dir = result_dir + '/pareto'
    entire_result_dir = result_dir + '/entire'

    if not os.path.exists(pareto_result_dir):
        os.mkdir(pareto_result_dir)

    if not os.path.exists(entire_result_dir):
        os.mkdir(entire_result_dir)

    result_dirs = [pareto_result_dir, entire_result_dir]

    # build the multi-objective optimization class
    # if you want to get the better results, you have to implement the preprocess function like surviving the better
    # individuals
    nsga2 = NSGA2(objective_indexes, objective_function_class, individual_num, epochs, optimal_dir,
                  preprocess_env_class, crossover_times, survival_coef=survival_coef, decline_step=decline_step,
                  gene_step=gene_step, elite_sort_objective_names=elite_sort_objective_names)

    # fixed dict
    fixed_dict = sga.set_fixed_dict()

    if sga.strategy == 'normal':
        initial_explore_type = False
        # run the multi-objective function
        nsga2.explore(fixed_dict, result_dirs, initial_explore_type=initial_explore_type)

    elif sga.strategy == 'onetime':

        nsga2.explore_onetime(fixed_dict, result_dirs)


if __name__ == '__main__':
    main_ga()



