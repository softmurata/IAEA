import time
from thermal_dp_cy import calc_design_point
from design_variable_cy import DesignVariable
from mission_tuning_cy import InitMission
from thermal_doff_cy import calc_off_design_point

def test_separate():

    aircraft_name = 'A320'
    engine_name = 'V2500'
    aircraft_type = 'normal'

    # data path
    aircraft_data_path = './DataBase/aircraft_test.json'
    engine_data_path = './DataBase/engine_test.json'
    mission_data_path = './Missions/fuelburn18000.json'

    data_base_args = [aircraft_data_path, engine_data_path, mission_data_path]

    # turbofan test
    # propulsion_type = 'turbofan'
    # dv_list = [4.7, 30.0, 1.61, 1380]

    # TeDP test
    propulsion_type = 'TeDP'
    dv_list = [40.0, 1430, 5.0, 1.24, 0.7, 0.99, 3]  # TeDP

    # Partial Electric test
    # propulsion_type = 'PartialElectric'
    # dv_list = [3.7,30.0,1.7,1380,5.0,1.5,0.9,0.99,3]#PartialElectric

    # build design variable class
    dv = DesignVariable(propulsion_type, aircraft_type)

    thermal_design_variables = dv.set_design_variable(dv_list)

    print(thermal_design_variables)

    # design point params
    design_point_params = [10668, 0.78]

    # init mission class
    init_mission_class = InitMission(aircraft_name, engine_name, aircraft_data_path, engine_data_path)

    # build calc design point class
    # cdp = CalcDesignPoint(aircraft_name, engine_name, aircraft_type, propulsion_type, thermal_design_variables, init_mission_class, design_point_params,
    #                      data_base_args)
    # cdp.run_dp()
    # cdp.objective_func_dp()

    cdp = calc_design_point(aircraft_name, engine_name, aircraft_type, propulsion_type, thermal_design_variables, init_mission_class, design_point_params, data_base_args)

    # off design point variables
    off_altitude = 0
    off_mach = 0
    off_required_thrust = 133000  # [N]

    off_param_args = [off_altitude, off_mach, off_required_thrust]

    # revolve rate lp shaft and fan
    rev_lp = 1.37  # 1.391
    rev_fan = 1.0

    rev_args = [rev_lp, rev_fan]

    # define design off point class
    cdop = calc_off_design_point(aircraft_name, engine_name, aircraft_type, propulsion_type, thermal_design_variables,
                              off_param_args, design_point_params, data_base_args, cdp, rev_args)

    print('rotation percentage: {} [%]'.format(rev_lp * 100))
    # print('design distributed ratio: {}, off design distributed ratio: {}'.format(dv_list[4], cdop.doff[33]))



if __name__ == '__main__':
    start = time.time()
    test_separate()
    finish = time.time()

    print(finish - start, '[s]')