from .integration_env_explore import IntegrationEnvExplore
from .determine_engine_params import EngineTuning
from .air_shape import NormalShape
from DataBase.make_json_data import *

# confirmation
# the fuel weight consistence


# test code according to the tuning style
# 1. engine design variable tuning, aircraft design variable tuning and mission tuning
# 2. aircraft design variable tuning and mission tuning
# 3. engine design variable tuning and mission tuning
# 4. mission tuning
# 5. no tuning

# function for creating database
def make_database_file(baseline_args):
    (baseline_aircraft_name, baseline_aircraft_type), (baseline_engine_name, baseline_propulsion_type), \
    (aircraft_data_path, engine_data_path) = baseline_args
    print('From now on, you have to find and input the public values of following aircraft and engine')
    print('=' * 10 + 'CONFIGURATION' + '=' * 10)
    print('aircraft type:', baseline_aircraft_type)
    print('aircraft name:', baseline_aircraft_name)
    print('propulsion type:', baseline_propulsion_type)
    print('engine name:', baseline_engine_name)

    # before running this part, you have to confirm the contents of database files and recognize the shortage target
    print('Please input arrange type of database: Overall or Add')
    arrange_type = str(input())

    # Completely create
    if arrange_type == 'Overall':
        make_json_file(baseline_aircraft_name, baseline_engine_name, aircraft_data_path, engine_data_path)
    # Partially fixed
    elif arrange_type == 'Add':
        # Initialize additional indexes
        lack_names = []

        print('=' * 5 + 'EXPLANATION OF ADDITIONAL MODE' + '=' * 5)
        print('Please input objective index name you want to add')
        print('Format of inserting is like: engine-engine_diameter, engine-weight, aircraft-yaw_moment')
        print('If you have no additional index, you have to input the word "finish" in order to quit this mode')
        print('=' * 20)

        while True:
            print('Please target name =>')
            target_name = str(input())

            if target_name == 'finish':
                break

            lack_names.append(target_name)

        for lack_index_name in lack_names:
            target_name = lack_index_name.split('-')[1]
            print('Please input {} value:'.format(target_name))

            lack_index_val = 0

            # In this case, we assume the propulsion type is turbofan
            # If you want to set another one as the baseline propulsion type, you have to change the dv_name list.
            # You should refer to the design variable class(design_variable.py) and fixed its list
            if target_name == 'design_variables':
                dv_list = []
                dv_name = None
                if baseline_propulsion_type in ['turbojet', 'turboshaft']:
                    dv_name = ['OPR', 'TIT']
                elif baseline_propulsion_type == 'turbofan':
                    dv_name = ['BPR', 'OPR', 'FPR', 'TIT']

                # If the dv_name is null, the operation finishes
                if dv_name is None:
                    exit()

                for dvn in dv_name:
                    print('now {} value'.format(dvn))
                    dvv = float(input())
                    dv_list.append(dvv)

                lack_index_val = dv_list

            else:

                lack_index_val = float(input())

            add_lack_data(baseline_aircraft_name, baseline_engine_name, lack_index_name, lack_index_val,
                          aircraft_data_path, engine_data_path)


def test_engine_aircraft_mission_tuning():
    # Global variables written by type of string are words you have to set before running code
    # data base path
    aircraft_data_path = './DataBase/aircraft_test.json'
    engine_data_path = './DataBase/engine_test.json'
    mission_data_path = './Missions/cargo1.0_passenger1.0.json'

    # set the data base args
    data_base_args = [aircraft_data_path, engine_data_path, mission_data_path]

    # baseline engine type and aircraft type
    baseline_aircraft_name = 'A320'  # A319, B737 and so on
    baseline_aircraft_type = 'normal'  # 'normal' or 'BWB'
    baseline_engine_name = 'V2500'  # 'V2500', 'CFM56' and so on
    baseline_propulsion_type = 'turbofan'  # 'turbojet', 'turbofan'

    # set the baseline arguments
    baseline_args = [(baseline_aircraft_name, baseline_aircraft_type), (baseline_engine_name, baseline_propulsion_type),
                     (aircraft_data_path, engine_data_path)]

    # Before conducting this exploration, we have to restore the public values into the database files by surveying the website
    make_database_file(baseline_args)

    # Survey target arguments
    current_aircraft_name = 'A320'
    current_aircraft_type = 'normal'
    current_propulsion_type = 'turbofan'
    current_engine_name = 'V2500'

    # set survey target argument
    current_args = [(current_aircraft_name, current_aircraft_type), (current_engine_name, current_propulsion_type),
                    (aircraft_data_path, engine_data_path)]

    # off design parameters
    off_altitude = 0.0  # [m]
    off_mach = 0.0
    off_required_thrust = 133000  # [N]

    # set the off parameters arguments
    off_param_args = [off_altitude, off_mach, off_required_thrust]

    # Build Overall Exploration class
    # constraint type
    constraint_type = 'TIT'
    # engine mounting positions (x_pos and y_pos)
    engine_mount_coef_x = 0.2
    engine_mount_coef_y = 0.2
    engine_mounting_positions = [engine_mount_coef_x, engine_mount_coef_y]

    # lift by drag calculation type
    ld_calc_type = 'constant-static'
    
    # Build explore class
    iee = IntegrationEnvExplore(baseline_args, current_args, off_param_args, mission_data_path, constraint_type,
                                engine_mounting_positions, ld_calc_type)

    # engine tuning and aircraft shape tuning <= iee.init()
    # ToDo check the performance of Gradient method, also we have to implement save and load tuning data method
    # engine_tuning_class = EngineTuning(baseline_aircraft_type, baseline_engine_name, baseline_aircraft_type,
    # baseline_propulsion_type, off_param_args, data_base_args)






if __name__ == '__main__':
    test_engine_aircraft_mission_tuning()




