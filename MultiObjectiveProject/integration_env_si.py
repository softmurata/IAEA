import numpy as np
from integration_env_explore import IntegrationEnvExplore

# for test
from design_variable import DesignVariable


class IntegrationEnvSwarm(IntegrationEnvExplore):
    """
    Attributes
    name: str
    design_count: int
                  the count which indicates the position of exploring
    """

    def __init__(self, baseline_args, current_args, off_param_args, mission_data_path, constraint_type, design_point,
                 engine_mounting_positions=[0.2, 0.2], ld_calc_type='constant-static'):

        super().__init__(baseline_args, current_args, off_param_args, mission_data_path, constraint_type, design_point,
                         engine_mounting_positions, ld_calc_type)

        self.name = 'Swarm'
        # the count of finishing calculating the design variables
        self.design_count = 0

    # calculate the performance for 1 design variables
    def fitness(self, thermal_design_variables):
        """

        :param thermal_design_variables: list
        :return: calc_results_dict: dict
                 the dictionary for swarm intelligence optimization
        """
        print('')
        print('')
        print('')
        print('')
        print('')
        print('Current Design Number: {}'.format(self.design_count))
        print('')
        print('')
        print('')
        print('')
        print('')
        # the shape of design variables is 'thermal'
        calc_results_dict = {}

        # meet the constraint
        if self.design_point == 'ground':
            calc_result_list = self.run_meet_thrust(thermal_design_variables, self.cruise_range)
        else:
            calc_result_list = self.run_meet_constraints(thermal_design_variables, self.cruise_range)

        self.design_count += 1

        # Error process
        if calc_result_list is None or len(calc_result_list) == 0:
            return None

        # insert the value
        for key, result in zip(self.results_list_index_dict.keys(), calc_result_list):

            calc_results_dict[key] = result

        return calc_results_dict


def test():
    # BaseLine Arguments
    baseline_aircraft_name = 'A320'
    baseline_aircraft_type = 'normal'
    baseline_engine_name = 'V2500'
    baseline_propulsion_type = 'turbofan'

    # data path
    aircraft_data_path = './DataBase/aircraft_test.json'
    engine_data_path = './DataBase/engine_test.json'
    mission_data_path = './Missions/cargo1.0_passenger1.0_.json'
    # mission_data_path = './Missions/cargo0.8_passenger1.0_.json'

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

    # engine mounting positions
    engine_mount_coef_x = 0.2
    engine_mount_coef_y = 0.2
    engine_mounting_positions = [engine_mount_coef_x, engine_mount_coef_y]

    # lift and drag calculation type
    ld_calc_type = 'constant-static'

    ies = IntegrationEnvSwarm(baseline_args, current_args, off_param_args, mission_data_path, constraint_type, design_point,
                              engine_mounting_positions, ld_calc_type)

    # confirmation for building class object
    # print(ies.total_mission_class.engine_weight_coef)
    # print(ies.init_mission_class.fuel_weight)

    # define fixed design variable
    # fixed_dict = {'OPR': 30, 'FPR': 1.66}
    fixed_dict = {'OPR': 30}
    individual_num = 50
    epochs = 2

    # run random overall exploration
    # generate initial design variable collection
    dv = DesignVariable(current_propulsion_type, current_aircraft_type)
    dv.generate_si_design_variable_collect(individual_num, fixed_dict)
    design_variables_collect = dv.therm_design_variable_collect

    for epoch in range(epochs):
        objective_func_values, next_design_variables_collect = ies.swarm_replay(design_variables_collect)
        design_variables_collect = next_design_variables_collect


if __name__ == '__main__':
    test()
