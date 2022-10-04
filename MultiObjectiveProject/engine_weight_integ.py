import numpy as np
import time
from design_variable_cy import DesignVariable
from mission_tuning_cy import InitMission
from thermal_dp_cy import calc_design_point
from thermal_doff_cy import calc_off_design_point
from engine_weight_cy import calc_engine_weight


start = time.time()
aircraft_name = 'A320'
engine_name = 'V2500'
aircraft_type = 'normal'
propulsion_type = 'TeDP'

dv_list = [30.0, 1380, 4.7, 1.61, 0.6, 0.99, 3]

# Set the thermal design variables
dv = DesignVariable(propulsion_type, aircraft_type)

thermal_design_variables = dv.set_design_variable(dv_list)

# database path
engine_data_path = './DataBase/engine_test.json'
aircraft_data_path = './DataBase/aircraft_test.json'
mission_data_path = './Missions/fuelburn18000.json'
data_base_args = [aircraft_data_path, engine_data_path, mission_data_path]

# off design parameters
off_altitude = 0.0
off_mach = 0.0
off_required_thrust = 133000  # [N]
off_param_args = [off_altitude, off_mach, off_required_thrust]

# design point params
design_point_params = [10668, 0.78]

# build initial mission class
init_mission_class = InitMission(aircraft_name, engine_name, aircraft_data_path, engine_data_path)

# build design point class
cdp = calc_design_point(aircraft_name, engine_name, aircraft_type, propulsion_type, thermal_design_variables,
                          init_mission_class, design_point_params, data_base_args)

# revolve rate
rev_lp = 1.4
rev_fan = 1.0
rev_args = [rev_lp, rev_fan]

# build off design point class
cdop = calc_off_design_point(aircraft_name, engine_name, aircraft_type, propulsion_type, thermal_design_variables,
                              off_param_args, design_point_params, data_base_args, cdp, rev_args)


# build engine weight class
args = [aircraft_name, engine_name, aircraft_type, propulsion_type, engine_data_path, cdp, cdop]

ew = calc_engine_weight(args)

finish = time.time()

print(finish - start)

