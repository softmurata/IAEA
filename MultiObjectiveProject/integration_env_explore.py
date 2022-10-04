import numpy as np
import json
import time
import os
# from thermal_doff import CalcDesignOffPoint
from thermal_doff_cy import calc_off_design_point
from thermal_dp_cy import calc_design_point
from engine_weight_cy import calc_engine_weight
from air_shape_cy import calc_normal_airshape, calc_bwb_airshape
# from engine_weight import EngineWeight
from mission_tuning import InitMission
from total_mission_tuning import TotalMission
from constraints import EnvConstraint
from design_variable_cy import DesignVariable
from air_shape import *
from flight_simu import FlightSimulation

# Build class order
# Define mission
# if required, tune params of engine(determine_engine_params.py),
# tune weight coefficients and length coefficients(total_mission_tuning.py)
# Set design variable
# Calculate on design point and off design point
# Calculate engine weight, aircraft weight and electric equipment weight
# Calculate Flight simulation under the particular conditions and get Fuel Consumption


class IntegrationEnvExplore(object):
    """
    Note:

        ############  Setting Part #############
        0. determine the combination of baseline aircraft and engine

        1. set element efficiency of engine components and design variables(mainly engine configuration)

           ex) bypass ratio(BPR), overall pressure ratio(OPR), fan pressure ratio(FPR), turbine inlet temperature(TIT)
           , the number of stages of each engine components(low pressure compressor, high pressure compressor, combustion chamber, high pressure turbine, low pressure turbine, fan, distributed fan, afterburner, jetnozzle)

        #############  Design Point Part #############

        2. assume airflow per time and flight condition(altitude, mach, required thrust)
           In latter chapter, flight condition is thought of as design point

        3. calculate physical features at the design point
           ex) air density, static temperature, static pressure

        4. execute cycle analysis at the design point and extract important objective indexes (specific fuel consumption, specific thrust)


        ############ Off Design point Part ###############

        5. assume the same element efficiency and design variables as those at the design point and set the constraint condition
           ex) Turbine Inlet Temperature at off design point(TIT), Temperature of final stage of compressor(COT), Blade height of final srage of 

        6. assume the revolving ratio of low pressure and high pressure rotational speed

        7. execute cycle analysis and confirm the energy matching at both low pressure side and high pressure one

        ########### Estimation of weight ###########

        8. calculate engine weight by the theory of NASA
           8.1 determine each cross sectional area
           8.2 calculate diameter and length individually by cross sectional area
           8.3 set the stage load coefficient against each components and decide the number of stage
           8.4 diameter, length, the stage number => estimate engine weight

        9. calculate aircraft weight
           9.1 set parameters (aspect ratio, chord span, thickness and chord span ratio, tip hub ratio, the edge and the back of angle)
           9.2 determine fuselage shape
           9.3 determine wing airfoil
           9.4 estimate the aircraft weight including the effect of various things
               (reference Aircraft Design Approach)


        ########### Flight Path Part #########

        10. according to the baseline type, the ratio of required thrust and max takeoff weight has to be adjusted and target required thrust is calculated

        11. confirm whether thrust and constraint index meet requirements

        12. determine the value of lift by drag according to flight path
            if necessary, applying the theory based on fluid dynamic in order to acquire lift by dtag

        13. calculate fuel burn by Breguar equations



    Attributes
    -------------------

    name: str
          environmental name
    baseline_aircraft_type: str
              aircraft type before investigation of target integration system
              for example, 'normal'
    baseline_aircraft_name: str
              aircraft name before investigation of target integration system
              for example, 'A320'
    baseline_propulsion_type: str
              type of target propulsion system
              for example, 'turbofan'
    baseline_engine_name: str
              name of target propulsion system (another call is engine)
              for example, 'V2500'
    aircraft_name: str
              name of aircraft under investigation of optimization and computation of performances
    aircraft_type: str
              type of aircraft under investigation of optimization and computation of performances
    engine_name: str
              name of engine under investigation of optimization and computation of performnaces
    propulsion_type: str
              type of propulsion system under investigation of optimization and computation of performances
    aircraft_data_path: str
              the relative path of restoring the data of aircraft's configurations
              the public values are in the files
    engine_data_path: str
              the relative path of restoring the data of engine's configurations
    data_base_args: list
              the list of database paths => [aircraft_data_path, engine_data_path, mission_data_path]
    off_param_args: list
              the list of off design parameters => [off_altitude, off_mach, off_required_thrust]
              if necessary, off_required_thrust is capable of being estimated by coefficient of thrust and takeoff weight
              at the requirement
    engine_mounting_positions: list
              the position of engine attaching to the aircraft => [engine_mounting_position_x, engine_mounting_position_y]
    isbuildmission: boolean
              flag of indicating the presence of establishment of missions
    init_mission_class: class object
              the class object which includes the values of mission under investigation
    total_mission_class: class object
              the class object which determine fixed coefficients of systems such as aircraft or engine in order to
              match the public values like websites
    isbuildshape: boolean
              flag of indicating the presence of tuning the baseline aircraft shape so that the wet area is equal
              to the target one
    env_constraints: class object
              the class object which keep the data of constraint condition
    constraint_type: str
              the type of constraint condition, for example 'TIT'
    constraint_target: float
              the value of constraint condition
    design_variable: class object
              the class object which conducts various operations on the design variables
              for instance, generate the collections for Swarm intelligence or change them into
              the computational environment for thermal and aerodynamic performances
    design_variables_num: int
              the number of design variables
    flight_simu: class object
              the class object which contains the method of calculating various objective indexes with setting flight path
    epsconv: float
              the epsilon for convergence flag
    result_list_index_dict: dict
              the dictionary which has the combinations of names of objective indexes and number index of them

    """

    def __init__(self, baseline_args, current_args, off_param_args, mission_data_path, constraint_type, design_point,
                 engine_mounting_positions, ld_calc_type):
        """

        :param baseline_args: baseline aircraft and engine data path

        :param current_args: current aircraft and engine data path [(name, type)]

        :param off_param_args: [off altitude, off mach number, off required thrust]

        :param mission_data_path: data path of missions

        :param constraint_type: set the constraint type (ex. TIT)
        """
        self.name = 'OverallExplore'

        self.design_point, self.design_point_params = design_point
        self.dp_params = design_point

        # set baseline arguments
        baseline_aircraft_args, baseline_engine_args, baseline_data_path_args = baseline_args
        self.baseline_aircraft_name, self.baseline_aircraft_type = baseline_aircraft_args
        self.baseline_engine_name, self.baseline_propulsion_type = baseline_engine_args

        # set current arguments
        aircraft_args, engine_args, data_path_args = current_args

        self.aircraft_name, self.aircraft_type = aircraft_args
        self.engine_name, self.propulsion_type = engine_args
        self.aircraft_data_path, self.engine_data_path = data_path_args

        self.data_base_args = [self.aircraft_data_path, self.engine_data_path, mission_data_path]

        # off design point parameters
        self.off_param_args = off_param_args

        # engine mounting positions for aircraft
        self.engine_mounting_positions = engine_mounting_positions

        # if you want to tune the design variables of engine, you have to implement the engine tuning class
        # after this part
        #####################

        #####################

        # Whether mission already exists or not
        self.isbuildmission = False

        # define mission
        self.init_mission_class, self.total_mission_class = self.define_mission(baseline_args, off_param_args,
                                                                                mission_data_path, design_point)
        self.mission_data_path = mission_data_path

        # define baseline aircraft shape
        self.isbuildshape = False

        # if you want to construct complete mission, baseline_tuning is True
        self.baseline_tuning = False

        if self.baseline_tuning:
            self.define_baseline_aircraft_shape()

        # define constraint class
        self.env_constraints = EnvConstraint()
        # constraint type
        # TIT, COT, compressor_out_blade_height, front_diameter, wide_length_distributed_fan
        self.constraint_type = constraint_type
        self.constraint_target = self.env_constraints.get_constraint_target(self.constraint_type)

        # build design variable class
        self.design_variable = DesignVariable(self.propulsion_type, self.aircraft_type)

        # design variables num
        self.design_variables_num = len(self.design_variable.design_variable_name)

        # simulation convergence target
        self.epsconv = 1.0e-4  # comparison to turbofan, loose the limit  : 1.0e-5

        self.design_point = design_point

        self.ld_calc_type = ld_calc_type

        self.isrange = False
        # local variables for cruise tuning
        self.cruise_range = 4500
        # switching boolean for cruise tuning
        self.tuning_range = False


        # results list index names
        self.results_list_index_dict = {'fuel_weight': 0, 'engine_weight': 1, 'aircraft_weight': 2,
                                        'max_takeoff_weight': 3, 'electric_weight': 4, 'isp': 5,
                                        'sfc': 6, 'co_blade_h': 7, 'th_ew_ratio': 8}

    def build_baseline_mission_config(self, baseline_args, off_param_args, mission_data_path, design_point):
        """

        :param baseline_args: [aircraft_name, aircraft_type]
        :param off_param_args: [off_altitude: float, off_mach: float, off_required_thrust: float]
        :param mission_coef_args: [cargo_coef: float => (0, 1), passenger_coef: float => (0, 1)]
        :param mission_data_path: str
        :return: init_mission_class, total_mission_class
        """

        # baseline mission class
        # set baseline args
        baseline_aircraft_args, baseline_engine_args, data_path_args = baseline_args

        baseline_aircraft_name, baseline_aircraft_type = baseline_aircraft_args
        baseline_engine_name, baseline_propulsion_type = baseline_engine_args
        aircraft_data_path, engine_data_path = data_path_args

        init_mission_class = InitMission(baseline_aircraft_name, baseline_engine_name, aircraft_data_path,
                                         engine_data_path)

        init_mission_class.set_maxpayload_mission(design_point)
        init_mission_class.load_mission_config(mission_data_path)

        total_mission_class = TotalMission(baseline_args, self.dp_params, off_param_args, mission_data_path)

        return init_mission_class, total_mission_class

    def define_mission(self, baseline_args, off_param_args, mission_data_path, design_point):
        """

        :param baseline_args: [aircraft_name, aircraft_type]
        :param off_param_args: [off_altitude: float, off_mach: float, off_required_thrust: float]
        :param mission_data_path: str
        :return: init_mission_class, total_mission_class
        """
        if mission_data_path == './Missions/maxpayload_base.json' or mission_data_path == './Missions/maxpayload_bse_test.json':
            self.isbuildmission = False

        else:
            self.isbuildmission = True

        # set mission if the mission file is not
        if not self.isbuildmission:
            print('Required tuning')

            init_mission_class, total_mission_class = self.build_baseline_mission_config(baseline_args,
                                                                                         off_param_args,
                                                                                         mission_data_path,
                                                                                         design_point)
            total_mission_class.run_tuning()

        else:
            print('no required tuning')
            # build initial mission class
            init_mission_class = InitMission(self.aircraft_name, self.engine_name, self.aircraft_data_path,
                                             self.engine_data_path)

            target_fuelburn = float(mission_data_path.split('/')[-1].split('.')[0][8:])
            init_mission_class.target_fuelburn(target_fuelburn, mission_data_path)

            # save results
            f = open(mission_data_path, 'r')
            file = json.load(f)

            if design_point == 'ground':
                file['altitude'] = 0
                file['mach'] = 0
                file['required_thrust'], file['required_thrust_ground'] = file['required_thrust_ground'], file['required_thrust']

            f = open(mission_data_path, 'w')
            json.dump(file, f)
            f.close()

            init_mission_class.load_mission_config(mission_data_path)

            # build total mission class
            total_mission_class = TotalMission(baseline_args, self.dp_params, off_param_args, mission_data_path)
            total_mission_class.load_coef_config()

        return init_mission_class, total_mission_class

    # define the baseline aircraft shape
    def define_baseline_aircraft_shape(self):
        # confirm the theta front, theta back and ts coef in the aircraft data path
        f = open(self.aircraft_data_path, 'r')
        aircraft_data_file = json.load(f)[self.aircraft_name]

        if 'theta_back' in aircraft_data_file.keys():
            self.isbuildshape = True

        # if mission coefficients has been loaded from mission data file
        if self.total_mission_class.isload:
            # determine the design variables in order to meet the design requirements
            self.total_mission_class.judge_thrust_restrict()
            # build the engine weight class
            args = [self.baseline_aircraft_name, self.baseline_engine_name, self.baseline_aircraft_type, self.baseline_propulsion_type, self.engine_data_path, self.total_mission_class.calc_design_point_class, self.total_mission_class.calc_off_design_point_class]
            engine_weight_class = calc_engine_weight(args)
            # engine_weight_class = EngineWeight(self.baseline_aircraft_name, self.baseline_engine_name, self.baseline_aircraft_type, self.baseline_propulsion_type, self.engine_data_path, self.total_mission_class.calc_design_point_class, self.total_mission_class.calc_off_design_point_class)
            # create the engine weight class which has already been built
            # engine_weight_class.run_engine()
        else:
            engine_weight_class = self.total_mission_class.ew
        engine_amplitude = 1.0
        normal_shape_class = NormalShape(self.baseline_aircraft_name, self.init_mission_class, engine_weight_class,
                                         self.engine_mounting_positions, engine_amplitude)
        # tuning
        tuning_flag = False if self.isbuildshape else True
        normal_shape_class.run_airshape(self.aircraft_data_path, drawing=False, tuning=tuning_flag)

    # helper function of run_meet constraints
    # judge convergence
    def judge_convergence(self, convergence_index, th_coef, rev_lp):
        """

        :param convergence_index: the value of target convergence index
        :param th_coef: the ratio of current thrust and target thrust
        :param rev_lp: the revolving ratio at off design point against the one at the design point
        :return: conv_target_res: difference between current value of the convergence index and target value

                 convergence_flag: the state of checking convergence
        """
        # coefficient of convergence index
        convergence_coef = convergence_index / self.constraint_target
        # print('')
        # print('convergence coefficient:', convergence_coef)
        # print('')

        # residual of convergence target
        conv_target_res = 1.0 - convergence_coef
        # confirmation
        # print('convergence target residual:', conv_target_res)

        # flag of convergence
        convergence_flag = 'continue'

        # Success
        if abs(conv_target_res) < self.epsconv and th_coef >= 1.0:
            convergence_flag = 'success'

        # ToDO Additional casting case (Fuck case)
        if convergence_coef <= 0.9 and th_coef >= 1.0:
            print('-' * 5 + ' Fuck case ' + '-' * 5)
            convergence_flag = 'failure'

        # failure
        if abs(conv_target_res) < self.epsconv and th_coef < 1.0:
            print('-' * 3 + ' Thrust is not enough ' + '-' * 3)
            convergence_flag = 'failure'

        if rev_lp < self.env_constraints.rev_lp_min or rev_lp > self.env_constraints.rev_lp_max:
            print('-' * 5 + ' LP shaft revolve range over ' + '-' * 5)
            convergence_flag = 'failure'

        return conv_target_res, convergence_flag

    def run_meet_constraints(self, thermal_design_variables, cruise_range):
        """
        :param: thermal_design_variables: design variables for calculating thermal and aerodynamic performances of engine and aircraft
        :return: results_list: the list which contains objective indexes that is necessary for preparing final results
        """
        # for error process
        nan_count = 0

        # determine the convergence index and calculate continuously while converge

        conv_target_res = 0.0
        conv_target_resold = 0.0
        rev_lp = 1.1
        rev_lp_step = 0.02
        rev_fan = 1.0  # default
        iterloop = 0  # calculation count
        ref_thrust_cruise = self.init_mission_class.required_thrust

        if self.isrange:
            aircraft_name = self.baseline_aircraft_name
            engine_name = self.baseline_engine_name
            aircraft_type = self.baseline_aircraft_type
            propulsion_type = self.baseline_propulsion_type
        else:
            aircraft_name = self.aircraft_name
            engine_name = self.engine_name
            aircraft_type = self.aircraft_type
            propulsion_type = self.propulsion_type

        while True:
            # define revolve rate arguments
            rev_args = [rev_lp, rev_fan]
            iterloop += 1

            # build calc design point class
            tuning_args = [ref_thrust_cruise, self.tuning_range]
            cdp = calc_design_point(aircraft_name, engine_name, aircraft_type, propulsion_type, thermal_design_variables, self.init_mission_class, self.design_point_params, self.data_base_args, tuning_args)


            # build calc off design point class
            cdop = calc_off_design_point(aircraft_name, engine_name, aircraft_type, propulsion_type,
                                      thermal_design_variables, self.off_param_args, self.design_point_params, self.data_base_args, cdp, rev_args)

            # build engine weight class
            args = [aircraft_name, engine_name, aircraft_type,
                                               propulsion_type, self.engine_data_path, cdp, cdop]
            engine_weight_class = calc_engine_weight(args)

            stage_numbers = engine_weight_class.stage_numbers

            # calculate engine amplitude
            engine_amplitude = engine_weight_class.total_engine_weight / self.init_mission_class.engine_weight
            other_args = engine_amplitude

            # ToDo calculate air shape class
            air_shape_class = None
            if aircraft_type == 'normal':
                air_shape_class = calc_normal_airshape(self.aircraft_name, self.aircraft_data_path,
                                                        self.init_mission_class, engine_weight_class,
                                                        self.engine_mounting_positions, other_args)
            elif aircraft_type == 'BWB':
                init_shape_params = 0.8
                air_shape_class = calc_bwb_airshape(self.aircraft_type, self.init_mission_class,
                                                    thermal_design_variables, init_shape_params,
                                                    self.data_base_args,
                                                    other_args)

            # Sizing
            # build flight simulation class
            self.flight_simu = FlightSimulation(aircraft_name, aircraft_type, self.init_mission_class,
                                                self.total_mission_class, self.data_base_args, self.ld_calc_type)

            thrust_target, aircraft_weight, engine_weight, max_takeoff_weight_new, weight_fracs_allprod, \
            converge_flag, co_blade_h, electric_weight, cruise_lift_by_drag, mass_product, cruise_range = self.flight_simu.sizing_of_thrust_mtow_ratio(
                engine_weight_class, air_shape_class, thermal_design_variables, cruise_range)

            # Arrange the index for comparison
            thrust_off = cdop.thrust_off  # thrust at off design point
            tit_off = cdop.TIT  # Turbine Inlet Temperature at off design point
            cot_off = cdop.COT  # Compressor Out Temperature at off design point
            sfc_off = cdop.sfc_off  # specific fuel consumption at off design point
            fpr_off = cdop.fpr_off
            ans_rev_lp = cdop.rev_lp
            front_diameter = engine_weight_class.front_diameter  # core front diameter
            wide_dist_fan_length = engine_weight_class.distributed_fan_width_length  # wide length of distributed fan
            fpre_off = cdop.doff[32]  # Distributed fan ratio at off design point


            ############# Error Process ###############
            error_flag = False  # whether or not error exists in this method

            th_coef = thrust_off / thrust_target
            print('')
            print('Thrust ratio:', th_coef)
            print('')

            # ratio of required thrust at ground and engine weight
            th_ew_ratio = thrust_off / 9.8 / engine_weight

            print('Ratio of thrust at ground and engine weight:', th_ew_ratio)
            print('')

            # if thrust is not full, calculation is forced to finish
            if th_coef < 0.6:
                print('-' * 5 + ' Thrust is not enough!! ' + '=' * 5)
                error_flag = True

            # if the results of thrust diffuses, calculation is forced to finish
            if np.isnan(thrust_off):
                print('-' * 5 + ' Thrust is unable to calculate!! ' + '-' * 5)
                error_flag = True

            # if the sizing fails, calculation is force to finish
            if converge_flag or thrust_target == self.init_mission_class.required_thrust_ground:
                print('-' * 5 + ' Sizing failed!! ' + '-' * 5)
                error_flag = True

            # if the value of fan pressure ratio is more than 1.7, finish
            if fpre_off > 1.9:

                print('-' * 5 + ' Tip velocity of Distributed fan is over !! ' + '-' * 5)
                error_flag = True

            if fpr_off > 1.9:
                print('-' * 5 + ' Tip Velocity of fan is over!! ' + '-' * 5)
                error_flag = True

            # if the value of nan_count is more than 20, finish
            if np.isnan(sfc_off):
                nan_count += 1

            if np.isnan(thrust_target):
                error_flag = True

            if nan_count >= 20:
                print('-' * 5 + ' Off design points could not be found!! ' + '-' * 5)
                error_flag = True

            # if ans_rev_lp < self.design_variable.lp_range[0] or ans_rev_lp > self.design_variable.lp_range[1]:
            #    print('-' * 5 + 'Range revolving is over' + '-' * 5)
            #    error_flag = True

            print('')
            print('=' * 5 + ' OPTIMIZATION CONFIGURATIONS ' + '=' * 5)
            print('cruise range [km]:', cruise_range)
            print('target thrust[N]:', thrust_target)
            print('thrust at off design point[N]:', thrust_off)

            # calculate fuel weight
            fuel_weight = (1.0 - weight_fracs_allprod) * max_takeoff_weight_new
            print('1. FuelBurn[kg]:', fuel_weight, ' 2. Target FuelBurn[kg]:', self.init_mission_class.fuel_weight)

            self.max_takeoff_weight = max_takeoff_weight_new
            thrust_cruise = (self.max_takeoff_weight * mass_product - self.init_mission_class.fuelburn_coef * fuel_weight) * 9.81 / cruise_lift_by_drag / self.init_mission_class.engine_num
            print('ref thrust at cruise point [N]:', ref_thrust_cruise, 'thrust at cruise [N]:', thrust_cruise)
            print('=' * 40)
            print('')
            ref_thrust_cruise = thrust_cruise

            if error_flag:
                return None

            ################ Veritify whether or not the design variable can meet the constraints ############

            # [fuel_weight, engine_weight, aircraft_weight, max_takeoff_weight, thermal_design_variables,
            # electric_weight]
            results_list = []

            # define convergence index
            convergence_index = 0  # Initialize convergence index
            if self.constraint_type == 'TIT':
                convergence_index = tit_off
            elif self.constraint_type == 'COT':
                convergence_index = cot_off
            elif self.constraint_type == 'compressor_out_blade_height':
                convergence_index = co_blade_h
            elif self.constraint_type == 'front_diameter':
                convergence_index = front_diameter
            elif self.constraint_type == 'width_length_distributed_fan':
                convergence_index = wide_dist_fan_length

            # judge convergence
            conv_target_res, convergence_flag = self.judge_convergence(convergence_index, th_coef, rev_lp)

            if convergence_flag == 'success':

                if not self.tuning_range:
                    if thrust_cruise > ref_thrust_cruise:
                        print('-' * 5 + 'Can not fly according to Braguer equations ' + '-' * 5)

                        return results_list



                print('')
                print('=' * 20 + ' FINAL RESULTS ' + '=' * 20)
                print('Revolving rate Core:', rev_lp, '  Fan:', rev_fan, 'distributed ratio:', cdop.doff[33])
                print('TIT:', tit_off, 'COT:', cot_off, 'Compressor out blade height:', co_blade_h)
                print('Distributed fan wide length:', wide_dist_fan_length)
                print('=' * 60)
                print('')

                results_list = [fuel_weight, engine_weight, aircraft_weight, max_takeoff_weight_new, electric_weight,
                                engine_weight_class.isp, engine_weight_class.sfc, co_blade_h, th_ew_ratio]

                break

            if convergence_flag == 'failure':
                break

            if convergence_flag == 'continue':

                if conv_target_res * conv_target_resold < 0.0:
                    rev_lp_step *= 0.5

                # update lp shaft revolve rate
                rev_lp += np.sign(conv_target_res) * rev_lp_step

                conv_target_resold = conv_target_res

        return results_list

    # design point is at the ground
    def run_meet_thrust(self, thermal_design_variables, cruise_range):
        # for error process
        nan_count = 0

        # determine the convergence index and calculate continuously while converge

        conv_target_res = 0.0
        conv_target_resold = 0.0
        rev_lp = 1.0
        rev_lp_step = 0.01
        rev_fan = 1.0  # default
        iterloop = 0  # calculation count
        ref_thrust_cruise = self.init_mission_class.required_thrust

        # set component types
        if self.isrange:
            aircraft_name = self.baseline_aircraft_name
            engine_name = self.baseline_engine_name
            aircraft_type = self.baseline_aircraft_type
            propulsion_type = self.baseline_propulsion_type
        else:
            aircraft_name = self.aircraft_name
            engine_name = self.engine_name
            aircraft_type = self.aircraft_type
            propulsion_type = self.propulsion_type


        while True:
            # define revolve rate arguments
            rev_args = [rev_lp, rev_fan]
            iterloop += 1

            # build calc design point class
            cdp = calc_design_point(aircraft_name, engine_name, aircraft_type, propulsion_type, thermal_design_variables, self.init_mission_class, self.design_point_params, self.data_base_args)
            # 19180.253582176872
            if not self.tuning_range:
                # set cruise thrust
                cdp.dref[10] = ref_thrust_cruise
            else:
                cdp.dref[10] = self.init_mission_class.required_thrust

            # build calc off design point class
            cdop = calc_off_design_point(aircraft_name, engine_name, aircraft_type,
                                      propulsion_type,
                                      thermal_design_variables, self.off_param_args, self.design_point_params,
                                      self.data_base_args, cdp, rev_args)

            # build engine weight class
            args = [aircraft_name, engine_name, aircraft_type,
                    propulsion_type, self.engine_data_path, cdp, cdop]
            engine_weight_class = calc_engine_weight(args)

            # information of engine stage number
            stage_numbers = engine_weight_class.stage_numbers

            # calculate engine amplitude
            engine_amplitude = engine_weight_class.total_engine_weight / self.init_mission_class.engine_weight
            other_args = engine_amplitude


            air_shape_class = None
            if self.aircraft_type == 'normal':

                air_shape_class = calc_normal_air_shape(self.aircraft_name, self.aircraft_data_path, self.init_mission_class, engine_weight_class,
                                              self.engine_mounting_positions, other_args)
                # air_shape_class = NormalShape(self.aircraft_name, self.init_mission_class, engine_weight_class,
                #                            self.engine_mounting_positions, other_args)
                # air_shape_class.run_airshape(self.aircraft_data_path, drawing=False, tuning=False)

            elif self.aircraft_type == 'BWB':
                init_shape_params = 0.8
                air_shape_class = calc_bwb_airshape(self.aircraft_type, self.init_mission_class,
                                                       thermal_design_variables, init_shape_params,
                                                       self.data_base_args,
                                                       other_args)
                # air_shape_class = BlendedWingBodyShape(self.aircraft_type, self.init_mission_class,
                #                                        thermal_design_variables, init_shape_params,
                #                                        self.data_base_args,
                #                                        other_args)
                # air_shape_class.run_airshape()

            # Sizing
            thrust_target, aircraft_weight, engine_weight, max_takeoff_weight_new, weight_fracs_allprod, \
            converge_flag, co_blade_h, electric_weight, cruise_lift_by_drag, mass_product, cruise_range = self.flight_simu.sizing_of_thrust_mtow_ratio_at_high(
                    engine_weight_class, air_shape_class, thermal_design_variables, cruise_range)

            # Arrange the index for comparison
            thrust_off = cdop.thrust_off  # thrust at off design point
            tit_off = cdop.TIT  # Turbine Inlet Temperature at off design point
            cot_off = cdop.COT  # Compressor Out Temperature at off design point
            sfc_off = cdop.sfc_off  # specific fuel consumption at off design point
            fpr_off = cdop.fpr_off
            ans_rev_lp = cdop.rev_lp
            front_diameter = engine_weight_class.front_diameter  # core front diameter
            wide_dist_fan_length = engine_weight_class.distributed_fan_width_length  # wide length of distributed fan
            fpre_off = cdop.doff[32]  # Distributed fan ratio at off design point

            ############# Error Process ###############
            error_flag = False  # whether or not error exists in this method

            th_coef = thrust_off / thrust_target
            print('')
            print('Thrust ratio:', th_coef)
            print('')

            # ratio of required thrust at ground and engine weight
            th_ew_ratio = thrust_off / 9.8 / engine_weight

            if np.isnan(engine_weight):
                print('-' * 5 + ' Engine is not designable! ' + '-' * 5)
                error_flag = True

            # if thrust is not full, calculation is forced to finish
            if th_coef < 0.5:
                print('-' * 5 + ' Thrust is not enough!! ' + '-' * 5)
                error_flag = True

            # if the results of thrust diffuses, calculation is forced to finish
            if np.isnan(thrust_off):
                print('-' * 5 + ' Thrust is unable to calculate!! ' + '-' * 5)
                error_flag = True

            # if the sizing fails, calculation is force to finish
            if converge_flag or thrust_target == self.init_mission_class.required_thrust_ground:
                print('-' * 5 + ' Sizing failed!! ' + '-' * 5)
                error_flag = True

            # if the value of nan_count is more than 20, finish
            if np.isnan(sfc_off):
                nan_count += 1

            if np.isnan(thrust_target):
                error_flag = True

            if nan_count >= 20:
                print('-' * 5 + ' Off design points could not be found!! ' + '-' * 5)
                error_flag = True

            # if ans_rev_lp < self.design_variable.lp_range[0] or ans_rev_lp > self.design_variable.lp_range[1]:
            #    print('-' * 5 + 'Range revolving is over' + '-' * 5)
            #    error_flag = True

            print('')
            print('=' * 5 + ' OPTIMIZATION CONFIGURATIONS ' + '=' * 5)
            print('target thrust[N]:', thrust_target)
            print('thrust at off design point[N]:', thrust_off)

            # calculate fuel weight
            fuel_weight = (1.0 - weight_fracs_allprod) * max_takeoff_weight_new
            print('1. FuelBurn[kg]:', fuel_weight, ' 2. Target FuelBurn[kg]:', self.init_mission_class.fuel_weight)
            self.max_takeoff_weight = max_takeoff_weight_new
            thrust_cruise = (self.max_takeoff_weight * mass_product - self.init_mission_class.fuelburn_coef * fuel_weight) * 9.81 / cruise_lift_by_drag / self.init_mission_class.engine_num

            print('=' * 40)
            print('')

            ref_thrust_cruise = thrust_cruise

            if error_flag:
                return None

            ################ Veritify whether or not the design variable can meet the constraints ############

            # [fuel_weight, engine_weight, aircraft_weight, max_takeoff_weight, thermal_design_variables,
            # electric_weight]
            results_list = []

            convergence_flag = 'continue'
            conv_target_res = 1.0 - th_coef

            if abs(conv_target_res) <= 1.0e-4:
                convergence_flag = 'success'

            if np.isnan(conv_target_res):
                convergence_flag = 'failure'

            if abs(conv_target_res) > 1.0e-4:
                convergence_flag = 'continue'

            if convergence_flag == 'success':

                if not self.tuning_range:
                    if thrust_cruise > ref_thrust_cruise:
                        print('-' * 5 + 'Can not fly according to Braguer equations!!' + '-' * 5)

                        return results_list

                print('')
                print('=' * 20 + ' FINAL RESULTS ' + '=' * 20)
                print('Revolving rate Core:', rev_lp, '  Fan:', rev_fan, 'distributed ratio:', cdop.doff[33])
                print('TIT:', tit_off, 'COT:', cot_off, 'Compressor out blade height:', co_blade_h)
                print('Distributed fan wide length:', wide_dist_fan_length)
                print('=' * 60)
                print('')

                results_list = [fuel_weight, engine_weight, aircraft_weight, max_takeoff_weight_new,
                                electric_weight, engine_weight_class.isp, engine_weight_class.sfc, co_blade_h,
                                th_ew_ratio]
                self.off_design_variables = cdop.off_design_variables

                break

            if convergence_flag == 'failure':
                break

            if convergence_flag == 'continue':

                if conv_target_res * conv_target_resold < 0.0:
                    rev_lp_step *= 0.5

                # update lp shaft revolve rate
                rev_lp += np.sign(conv_target_res) * rev_lp_step

                conv_target_resold = conv_target_res

        return results_list

    # Arrange ideal cruise distance in order to target fuelburn
    def range_tuning(self, target_design_variables):
        self.tuning_range = True
        # create thermal design variables
        si_design_variables_collect = [target_design_variables]
        self.design_variable.si_design_variable_collect = si_design_variables_collect
        self.design_variable.generate_therm_design_variable_collect()

        # prepare for two split method in order to acquire ideal cruise range
        cruise_range = self.init_mission_class.range
        range_step = 50
        range_diff = 0.0
        range_diffold = 0.0
        count = 0
        exception_count = 0

        import time
        self.constraint_target = 1820

        while True:

            for thermal_design_variables in self.design_variable.therm_design_variable_collect:
                if self.design_point == 'ground':
                    calc_result_lists = self.run_meet_thrust(thermal_design_variables, cruise_range)
                else:
                    calc_result_lists = self.run_meet_constraints(thermal_design_variables, cruise_range)

            if len(calc_result_lists) == 0:
                if exception_count == 0:
                    range_step = 40
                exception_count += 1

            max_takeoff_weight = self.max_takeoff_weight  # calc_result_lists[3]
            # calculate difference of max takeoff weight
            range_diff = 1.0 - max_takeoff_weight / self.init_mission_class.max_takeoff_weight

            for _ in range(5):
                print('')
            print('max takeoff weight diff:', range_diff, 'range:', cruise_range)
            print('max takeoff weight:', max_takeoff_weight)
            time.sleep(1)
            # check convergence

            if abs(range_diff) < 1.0e-6:
                print('ok')
                break

            if count == 300:
                break

            # reduce range step according to the degree of difference
            if range_diff * range_diffold < 0.0:
                range_step *= 0.5
            # update cruise range
            cruise_range += np.sign(range_diff) * range_step
            range_diffold = range_diff
            count += 1

        # save tuned ideal range in the mission file
        self.cruise_range = cruise_range
        f = open(self.mission_data_path, 'r')
        mission_file = json.load(f)
        mission_file['range'] = cruise_range

        tuning_indexes = ['air_weight_coef', 'engine_weight_coef', 'engine_axis_coef', 'engine_length_coef', 'fuelburn_coef']
        tuning_indexes_value = [self.total_mission_class.air_weight_coef, self.total_mission_class.engine_weight_coef, self.total_mission_class.engine_axis_coef, self.total_mission_class.engine_length_coef, self.total_mission_class.fuelburn_coef]

        for key, val in zip(tuning_indexes, tuning_indexes_value):
            mission_file[key] = val

        f.close()
        # write
        write_file = open(self.mission_data_path, 'w')
        json.dump(mission_file, write_file)
        write_file.close()

        self.tuning_range = False

    def set_range(self, range):

        self.cruise_range = range

        f = open(self.mission_data_path, 'r')
        mf = json.load(f)
        f.close()

        mf['range'] = range

        f = open(self.mission_data_path, 'w')
        json.dump(mf, f)
        f.close()

    def set_tech_lev(self, tech_level):

        f = open(self.mission_data_path, 'r')
        mf = json.load(f)
        f.close()

        mf['tech_lev'] = tech_level

        f = open(self.mission_data_path, 'w')
        json.dump(mf, f)
        f.close()

    def overall_explore(self, individual_num, fixed_dict):
        """

        :param individual_num: the number of the collections of design variables
        :param fixed_dict: the dictionary which shows the fixed index
        :return: None
        """
        # generate swarm intelligence design variables collect
        self.design_variable.generate_si_design_variable_collect(individual_num, fixed_dict)
        # convert swarm intelligence design variables collect into thermal design variables collect
        self.design_variable.generate_therm_design_variable_collect()

        objectives_m = [[] for _ in range(10)]

        for thermal_design_variables in self.design_variable.therm_design_variable_collect:
            print('')
            print('thermal design variables:')
            print(thermal_design_variables)
            print('')

            calc_results_list = self.run_meet_constraints(thermal_design_variables, self.cruise_range)

            if calc_results_list is None or len(calc_results_list) == 0:
                continue

            for idx, result in enumerate(calc_results_list):
                objectives_m[idx].append(result)

        print(objectives_m[0])


# range tuning test
def test_v2500():

    # display the docstring of class object
    # print(IntegrationEnvExplore.overall_explore.__doc__)
    # exit()

    # BaseLine Arguments
    baseline_aircraft_name = 'A320'
    baseline_aircraft_type = 'normal'
    baseline_engine_name = 'V2500'
    baseline_propulsion_type = 'turbofan'

    # data path
    aircraft_data_path = './DataBase/aircraft_test.json'
    engine_data_path = './DataBase/engine_test.json'
    # mission_data_path = './Missions/cargo1.0_passenger1.0_.json'
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
    current_propulsion_type = 'TeDP'

    current_args = [(current_aircraft_name, current_aircraft_type), (current_engine_name, current_propulsion_type),
                    (aircraft_data_path, engine_data_path)]

    # Build Overall Exploration class
    # constraint type
    constraint_type = 'TIT'

    # set the engine mounting positions
    engine_mount_coef_x = 0.2
    engine_mount_coef_y = 0.2
    engine_mounting_positions = [engine_mount_coef_x, engine_mount_coef_y]

    # lift by drag calculation type
    ld_calc_type = 'constant-static'

    design_point = ['cruise', [10668, 0.78]]

    range_tuning = False

    # maxpayload tuning part
    mission_data_path = './Missions/fuelburn18000.json'  # './Missions/maxpayload_base.json'
    if range_tuning:
        iee = IntegrationEnvExplore(baseline_args, baseline_args, off_param_args, mission_data_path, constraint_type, design_point,
                                    engine_mounting_positions, ld_calc_type)


        # confirmation for building class object
        print(iee.total_mission_class.engine_weight_coef)
        print(iee.init_mission_class.fuel_weight)

        # V2500 design variables
        v2500_design_variables = [4.7, 30.0, 1.61, 1380]

        iee.range_tuning(v2500_design_variables)

        ideal_range = iee.cruise_range

    ideal_range = 4808.0812087409595

    iee = IntegrationEnvExplore(baseline_args, current_args, off_param_args, mission_data_path, constraint_type,
                                design_point,
                                engine_mounting_positions, ld_calc_type)

    iee.set_range(ideal_range)
    print('ideal range:', ideal_range)

    start = time.time()

    # Turbofan
    # iee.overall_explore(1, fixed_dict={'BPR': 4.7, 'OPR': 30, 'FPR': 1.61, 'TIT': 1380})

    # test case
    # iee.overall_explore(1, fixed_dict={'OPR': 44, 'TIT': 1500, 'div_alpha': 0.8, 'BPRe': 7.4, 'FPRe': 1.27, 'nele': 0.99, 'Nfan': 4})

    # TeDP
    iee.overall_explore(1, fixed_dict={'OPR': 33, 'TIT': 1530, 'div_alpha': 0.8, 'BPRe': 16.1, 'FPRe': 1.3, 'nele': 0.99, 'Nfan': 3})

    finish = time.time()

    print('Computation time [s]:', finish - start)



if __name__ == '__main__':

    test_v2500()




