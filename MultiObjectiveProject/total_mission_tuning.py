from engine_weight import EngineWeight
from thermal_doff import CalcDesignOffPoint
from thermal_dp import CalcDesignPoint
from design_variable import DesignVariable
from mission_tuning import InitMission
from aircraft_weight import AircraftWeight
from constraints import EnvConstraint
import numpy as np
import json
import time


# Tuning fixed coefficient
class TotalMission(object):

    def __init__(self, baseline_args, design_point, off_param_args, mission_data_path):
        """

        :param baseline_args: [(baseline_aircraft_name, baseline_aircraft_type), (engine_name, propulsion_type),
        (aircraft_data_path, engine_data_path)]
        :param mission_data_path: str
        """

        aircraft_args, engine_args, data_path_args = baseline_args
        # set the aircraft data
        self.baseline_aircraft_name, self.baseline_aircraft_type = aircraft_args
        # set the engine data
        self.baseline_engine_name, self.baseline_propulsion_type = engine_args

        # set the data path
        self.baseline_aircraft_data_path, self.baseline_engine_data_path = data_path_args

        # set mission data path
        self.baseline_mission_data_path = mission_data_path

        # data base path arguments
        self.data_base_args = [self.baseline_aircraft_data_path, self.baseline_engine_data_path,
                          self.baseline_mission_data_path]

        # Already built initial mission tuning
        json_file = open(self.baseline_mission_data_path, 'r')
        # set mission file
        self.mission_file = json.load(json_file)

        # open files
        f = open(self.baseline_engine_data_path, 'r')
        engine_file = json.load(f)
        engine_file = engine_file[self.baseline_engine_name]

        f = open(self.baseline_aircraft_data_path, 'r')
        aircraft_file = json.load(f)
        aircraft_file = aircraft_file[self.baseline_aircraft_name]

        # Baseline thermal design variables
        # baseline design variable list
        baseline_dv_list = engine_file['design_variable']

        # build design variable class
        dv = DesignVariable(self.baseline_propulsion_type, self.baseline_aircraft_type)

        # Build baseline init mission class
        self.init_mission_class = InitMission(self.baseline_aircraft_name, self.baseline_engine_name,
                                              self.baseline_aircraft_data_path, self.baseline_engine_data_path)
        self.init_mission_class.load_mission_config(mission_data_path)

        # thermal design variable
        self.baseline_thermal_design_variables = dv.set_design_variable(baseline_dv_list)

        # Baseline engine weight
        self.baseline_engine_thrust = engine_file['required_thrust']
        self.baseline_engine_weight = engine_file['engine_weight']
        self.baseline_engine_diameter = engine_file['engine_diameter']
        self.baseline_engine_length = engine_file['engine_length']

        # Baseline aircraft_weight
        self.baseline_aircraft_weight = aircraft_file['aircraft_weight']
        self.baseline_max_takeoff_weight = aircraft_file['max_takeoff_weight']

        # design point arguments
        self.design_point, self.design_point_params = design_point

        # off parameters arguments
        self.off_param_args = off_param_args

        # indexes for requiring tuning
        self.tuning_indexes = ['air_weight_coef', 'engine_weight_coef', 'engine_axis_coef', 'engine_length_coef', 'fuelburn_coef']

        self.air_weight_coef = None

        self.engine_weight_coef = None

        self.engine_axis_coef = None

        self.engine_length_coef = None

        self.fuelburn_coef = None

        # load or not
        self.isload = True

        # gravity
        self.g = 9.81

        # TIT constraints
        self.ev = EnvConstraint()
        self.tit_constraint = self.ev.tit

    def run_cycle_tuning(self):

        self.judge_thrust_restrict()

    # Main function
    def run_tuning(self):
        """
        execute both engine tuning and aircraft tuning against the weight
        """

        self.judge_thrust_restrict()

        # Build EngineWeight Class
        self.ew = EngineWeight(self.baseline_aircraft_name, self.baseline_engine_name, self.baseline_aircraft_type,
                               self.baseline_propulsion_type, self.baseline_engine_data_path, self.calc_design_point_class, self.calc_off_design_point_class)


        self.engine_tuning()

        # Build Aircraft weight class
        engine_amplitude = 1.0
        aw_args = engine_amplitude
        self.aw = AircraftWeight(self.baseline_aircraft_name, self.baseline_aircraft_type, self.ew,
                                 self.init_mission_class, self.baseline_thermal_design_variables, self.data_base_args,
                                 aw_args)

        self.aircraft_tuning()

        save_path = './Missions/maxpayload_base.json'
        self.save_coef_config(save_path)

    # Judge whether the established engine can generate required thrust at off design point (In most cases, at ground)
    def judge_thrust_restrict(self):
        """
        explore the design point in order to meet the required thrust and determine the content of design variables
        """

        # while thrust meets required thrust at ground at the upper temperature
        # ToDo test this code

        # open baseline mission file
        f = open('./Missions/maxpayload_base.json', 'r')
        mission_file = json.load(f)
        f.close()

        # Initialize fuelburn coefficient
        fuelburn_coef = mission_file['fuelburn_coef']
        fuelburn_coef_step = 0.05
        mass_product = np.prod(self.init_mission_class.mass_ratio[:2])

        resthrust = 0.0
        resthrustold = 0.0
        count = 0

        # Thrust Tuning
        while True:
            thrust = (self.baseline_max_takeoff_weight - fuelburn_coef * self.init_mission_class.fuel_weight) * self.g / self.init_mission_class.Lift_by_Drag / self.init_mission_class.engine_num

            rev_lp = 0.9
            rev_lp_step = 0.01

            restit = 0.0
            restitold = 0.0

            titcount = 0

            # build calc design point class
            self.calc_design_point_class = CalcDesignPoint(self.baseline_aircraft_name, self.baseline_engine_name,
                                                           self.baseline_aircraft_type, self.baseline_propulsion_type,
                                                           self.baseline_thermal_design_variables,
                                                           self.init_mission_class, self.design_point_params,
                                                           self.data_base_args)

            self.calc_design_point_class.run_dp()
            self.calc_design_point_class.objective_func_dp()

            # TIT tuning
            while True:

                rev_args = [rev_lp, 1.0]
                # build calculate off design point class
                self.calc_off_design_point_class = CalcDesignOffPoint(self.baseline_aircraft_name,
                                                                      self.baseline_engine_name,
                                                                      self.baseline_aircraft_type,
                                                                      self.baseline_propulsion_type,
                                                                      self.baseline_thermal_design_variables,
                                                                      self.off_param_args, self.design_point_params,
                                                                      self.data_base_args, self.calc_design_point_class)

                self.calc_off_design_point_class.run_off(rev_args)

                self.calc_off_design_point_class.objective_func_doff()

                # residual tit
                restit = 1.0 - self.calc_off_design_point_class.TIT / self.tit_constraint

                # print('restit:', restit)

                if abs(restit) < 1.0e-4:
                    break

                if restit * restitold < 0.0:
                    rev_lp_step *= 0.5

                restitold = restit

                # update revolve lp shaft
                rev_lp += np.sign(restit) * rev_lp_step

                titcount += 1

                if titcount == 150:
                    exit()

            # residual of thrust

            resthrust = 1.0 - self.calc_off_design_point_class.thrust_off / self.init_mission_class.required_thrust_ground

            print('resthrust:', resthrust, 'thrust:', thrust, 'fuelburn coef:', fuelburn_coef)
            time.sleep(0.5)

            if abs(resthrust) < 1.0e-7 and resthrust < 0:
                break

            if resthrustold * resthrust < 0.0:
                fuelburn_coef_step *= 0.5

            resthrustold = resthrust

            fuelburn_coef += -np.sign(resthrust) * fuelburn_coef_step

            count += 1

            if count == 200:
                exit()

            # rewrite mission file
            file = open(self.baseline_mission_data_path, 'r')
            target_file = json.load(file)
            file.close()

            target_file['fuelburn_coef'] = fuelburn_coef
            file = open(self.baseline_mission_data_path, 'w')
            json.dump(target_file, file)
            file.close()

            #
            file = open('./Missions/maxpayload_base.json', 'r')
            target_file = json.load(file)
            file.close()

            target_file['fuelburn_coef'] = fuelburn_coef
            file = open('./Missions/maxpayload_base.json', 'w')
            json.dump(target_file, file)
            file.close()

        print('fuelburn coef:', fuelburn_coef)
        self.fuelburn_coef = fuelburn_coef


    def aircraft_tuning(self):
        """
        tuning against aircraft weight, in fact, get fixed coefficient for aircraft
        """

        self.aw.run_airframe()

        calc_airframe_weight = self.aw.weight_airframe

        self.air_weight_coef = self.baseline_aircraft_weight / calc_airframe_weight

    def engine_tuning(self):
        """
        tuning against engine weight, in fact, get fixed coefficient for engine
        """
        # calculate baseline engine by applying engine estimating model

        # Operate weight calculation
        self.ew.run_engine()

        # Compared indexes
        calc_engine_weight = self.ew.total_engine_weight
        calc_engine_length = self.ew.total_engine_length
        calc_engine_front_diameter = self.ew.front_diameter

        # calculate fix coefficients
        self.engine_weight_coef = self.baseline_engine_weight / calc_engine_weight
        # self.engine_axis_coef = self.baseline_engine_diameter / calc_engine_front_diameter
        # self.engine_length_coef = self.baseline_engine_length / calc_engine_length
        self.engine_axis_coef = 1.0
        self.engine_length_coef = 1.0

    # save weight coefficients configurations
    def save_coef_config(self, mission_data_path):
        """
        save fixed coefficients after tuning
        """

        target_indexes = [self.air_weight_coef, self.engine_weight_coef, self.engine_axis_coef, self.engine_length_coef, self.fuelburn_coef]

        for idx_n, ti in zip(self.tuning_indexes, target_indexes):
            self.mission_file[idx_n] = ti

        # Open in the state of writing
        file = open(mission_data_path, 'w')
        json.dump(self.mission_file, file)

        print('=' * 10 + ' Mission coefficient save success ' + '=' * 10)

    # load weight coefficients configurations
    def load_coef_config(self):
        """
        load fixed coefficients from previous tuning
        """
        f = open('./Missions/maxpayload_base.json')
        mission_file = json.load(f)
        self.air_weight_coef = mission_file['air_weight_coef']
        self.engine_weight_coef = mission_file['engine_weight_coef']
        self.engine_axis_coef = mission_file['engine_axis_coef']
        self.engine_length_coef = mission_file['engine_length_coef']

        print('=' * 10 + ' Mission coefficient load success ' + '=' * 10)


def test():

    # BaseLine Arguments
    baseline_aircraft_name = 'A320'
    baseline_aircraft_type = 'normal'
    baseline_engine_name = 'V2500'
    baseline_propulsion_type = 'turbofan'

    # data path
    aircraft_data_path = './DataBase/aircraft_test.json'
    engine_data_path = './DataBase/engine_test.json'
    mission_data_path = './Missions/maxpayload_base.json'

    baseline_args = [(baseline_aircraft_name, baseline_aircraft_type), (baseline_engine_name, baseline_propulsion_type),
     (aircraft_data_path, engine_data_path)]

    # off design parameters
    off_altitude = 0.0
    off_mach = 0.0
    off_required_thrust = 133000  # [N]

    off_param_args = [off_altitude, off_mach, off_required_thrust]

    # design point
    design_point = ['cruise', [10668, 0.78]]

    tm = TotalMission(baseline_args, design_point, off_param_args, mission_data_path)

    # tuning type
    tuning_type = 'weight'  # 'weight'

    if tuning_type == 'cycle':
        tm.run_cycle_tuning()

    elif tuning_type == 'weight':
        tm.run_tuning()

        print('engine weight coefficient:', tm.engine_weight_coef)
        print('axial engine coefficient:', tm.engine_axis_coef)
        print('total engine length coefficient:', tm.engine_length_coef)
        print('aircraft weight coefficient:', tm.air_weight_coef)


if __name__ == '__main__':
    test()
