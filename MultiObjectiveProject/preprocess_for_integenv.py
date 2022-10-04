from thermal_dp_cy import calc_design_point
from mission_tuning_cy import InitMission

# for test
from design_variable import DesignVariable


class PreprocessIntegrationEnv(object):

    def __init__(self, aircraft_name, engine_name, aircraft_type, propulsion_type, data_base_args, design_point):

        # Initialize required arguments
        self.aircraft_name = aircraft_name
        self.engine_name = engine_name
        self.aircraft_type = aircraft_type
        self.propulsion_type = propulsion_type
        self.data_base_args = data_base_args
        aircraft_data_path, engine_data_path, mission_data_path = data_base_args

        self.design_point, self.design_point_params = design_point

        self.init_mission_class = InitMission(self.aircraft_name, self.engine_name, aircraft_data_path, engine_data_path)

        # target the value of threshold
        self.obj_threshold = 0.8

    def select_better_individuals(self, design_variables_collect):

        # Initialize better design variables collection
        better_design_variables_collect = []

        for thermal_design_variables in design_variables_collect:
            tuning_args = [self.init_mission_class.required_thrust, False]
            # build the design point class
            calc_design_point_class = calc_design_point(self.aircraft_name, self.engine_name, self.aircraft_type,
                                                      self.propulsion_type, thermal_design_variables, self.init_mission_class, self.design_point_params,
                                                      self.data_base_args, tuning_args)

            # confirmation for test code
            # print('sfc:', calc_design_point_class.sfc)

            # check the requirements
            # this time, specific fuel consumption(sfc) is the target objectives
            if calc_design_point_class.sfc <= self.obj_threshold:
                better_design_variables_collect.append(thermal_design_variables)

        return better_design_variables_collect


# test code
def test():

    # global variables for normal exploration
    aircraft_name = 'A320'
    aircraft_type = 'normal'
    engine_name = 'V2500'
    propulsion_type = 'turbofan'

    # data base path
    aircraft_data_path = './DataBase/aircraft_test.json'
    engine_data_path = './DataBase/engine_test.json'
    mission_data_path = './Missions/cargo1.0_passenger1.0_.json'

    data_base_args = [aircraft_data_path, engine_data_path, mission_data_path]

    # build design variable class
    dv = DesignVariable(propulsion_type, aircraft_type)

    # swarm class local variables
    individual_num = 100
    fixed_dict = {'OPR': 30}

    # generate the design variables collection
    dv.generate_si_design_variable_collect(individual_num, fixed_dict)
    # change the form of design variables into the thermal performance's class
    dv.generate_therm_design_variable_collect()
    # set the design variables collection for another class
    current_design_variables_collect = dv.therm_design_variable_collect

    # build preprocess class
    pie = PreprocessIntegrationEnv(aircraft_name, engine_name, aircraft_type, propulsion_type, data_base_args)

    better_design_variables_collect = pie.select_better_individuals(current_design_variables_collect)

    print('survival individual num:', len(better_design_variables_collect))
    print('condition of design variables:', better_design_variables_collect[5])


if __name__ == '__main__':
    test()
