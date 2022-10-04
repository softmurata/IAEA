import numpy as np
import json
from AirComponent import *
from air_shape import BlendedWingBodyShape

# for test
from engine_weight import EngineWeight
from mission_tuning import InitMission
from thermal_doff import CalcDesignOffPoint
from thermal_dp import CalcDesignPoint
from design_variable import DesignVariable

# class for defining required components (Aircraft)
class InitAircraft(object):

    def __init__(self):

        self.normal_components = [MainWing, HorizontalWing, VerticalWing, Fuselage, MainLandingGear, NoseLandingGear,
                                  Nacelle, EngineControl, Starter, FuelSystem, FlightControl, APU, Instrument,
                                  Hydraulics, Electric, Avionics, Furnishing, AirConditioner, AntiIce, HandlingGear,
                                  PassengerEquip]

        self.bwb_components = [MainWing, HorizontalWing, Fuselage, MainLandingGear, NoseLandingGear,
                               Nacelle, EngineControl, Starter, FuelSystem, FlightControl, APU, Instrument,
                               Hydraulics, Electric, Avionics, Furnishing, AirConditioner, AntiIce, HandlingGear,
                               PassengerEquip]

    def get_component_classes(self, aircraft_type):

        selected_components = None

        if aircraft_type == 'normal':

            selected_components = self.normal_components

        elif aircraft_type == 'BWB':

            selected_components = self.bwb_components

        return selected_components


class AircraftWeight(object):

    def __init__(self, aircraft_name, aircraft_type, engine_weight_class, init_mission_class, thermal_design_variables, data_base_args,
                 other_args, init_shape_params=[0.5]):

        self.aircraft_name = aircraft_name
        self.aircraft_type = aircraft_type
        # init mission class (Already Build)
        self.init_mission_class = init_mission_class
        # engine_weight class
        self.engine_weight_class = engine_weight_class
        # degree of expansion of engine weight in response to baseline
        self.engine_amplitude = other_args
        # thermal design variables
        self.thermal_design_variables = thermal_design_variables
        # data base path
        self.data_base_args = data_base_args
        # initial shape params (for Blended Wing Body: cabin height ratio)
        self.init_shape_params = init_shape_params

        # set the results list
        self.weight_results = np.zeros(100)

        # total weight airframe
        self.weight_airframe = 0.0

        # unit change
        self.m_to_ft = 3.28084
        self.kg_to_lb = 2.204621

        # baseline max takeoff weight
        baseline_mission_data_path = './Missions/maxpayload_base.json'
        f = open(baseline_mission_data_path, 'r')
        mission_file = json.load(f)
        self.baseline_max_takeoff_weight = mission_file['max_takeoff_weight']

        # Build InitAir class
        ia = InitAircraft()
        self.aircraft_component_classes = ia.get_component_classes(aircraft_type)

        # build component classes
        self.build_aircraft_classes = []

        self.build_aircraft_component_classes()

        self.normal_components_names = ['MainWing', 'HorizontalWing', 'VerticalWing', 'Fuselage', 'MainLandingGear', 'NoseLandingGear',
                                  'Nacelle', 'EngineControl', 'Starter', 'FuelSystem', 'FlightControl', 'APU', 'Instrument',
                                  'Hydraulics', 'Electric', 'Avionics', 'Furnishing', 'AirConditioner', 'AntiIce', 'HandlingGear',
                                  'PassengerEquip']

    def build_aircraft_component_classes(self):

        for a_class in self.aircraft_component_classes:

            self.build_aircraft_classes.append(a_class(self.aircraft_name, self.aircraft_type,
                                                       self.init_mission_class,
                                                       self.engine_amplitude))

    # set new types of aircraft configuraton:
    def set_new_types_of_aircraft_config(self):

        if self.aircraft_type == 'BWB':
            # build Blended Wing Body class
            self.bwbs = BlendedWingBodyShape(self.aircraft_name, self.init_mission_class, self.thermal_design_variables,
                                             self.engine_weight_class,
                                             self.init_shape_params, self.data_base_args, self.engine_amplitude)
            # determine cabin shape
            self.bwbs.define_cabin_shape()
            # determine cross area distribution
            self.bwbs.compose_cross_area_dist()
            # determine main wing configuration
            self.bwbs.compose_main_wing_config()

    def run_airframe(self):

        # max takeoff weight
        max_takeoff_weight = self.baseline_max_takeoff_weight

        # set engine indexes (after calculation on and off design point)
        # engine_weight = self.engine_weight_class.total_engine_weight
        engine_weight = self.engine_weight_class.total_engine_weight
        front_diameter = self.engine_weight_class.front_diameter
        engine_length = self.engine_weight_class.total_engine_length

        # calculate sum of components
        for a_class in self.build_aircraft_classes:
            # print('')
            # print('component name: {} '.format(a_class.name))
            # print('')

            if a_class.name in ['Nacelle', 'Starter']:

                other_args = [engine_weight, front_diameter, engine_length]

            else:

                other_args = [max_takeoff_weight]

            # Blended Wing Body configuration
            if self.aircraft_type == 'BWB':
                self.set_new_types_of_aircraft_config()

                if a_class.name in ['Wing', 'Fuselage']:

                    a_class.set_bwb_config(self.bwbs)

            self.weight_results = a_class(self.weight_results, other_args)

        pre_weight_airframe = np.sum(self.weight_results)

        """
        # describe configuration for aircraft components
        print('')
        count = 0
        for idx in range(100):
            if self.weight_results[idx] == 0:
                continue
            name = self.normal_components_names[count]
            weight = self.weight_results[idx]
            count += 1
            print('{}: {}'.format(name, weight))
        """

        # electric weight
        electric_equip_weight = self.engine_weight_class.weight_results[99]
        electric_equip_weight *= self.engine_amplitude  # fix coefficients
        electric_equip_weight *= self.kg_to_lb

        # print('electric equipment weight:', electric_equip_weight)

        # add structure weight by additional electric equipment weight
        e_coef = (electric_equip_weight / pre_weight_airframe) * 0.0

        # Recalculate airframe weight
        self.weight_airframe = pre_weight_airframe * (1.0 + e_coef)

        # from lb to kg (unit change)
        self.weight_airframe = self.weight_airframe / self.kg_to_lb



def test():

    # global variables
    aircraft_name = 'A320'
    engine_name = 'V2500'
    aircraft_type = 'normal'
    propulsion_type = 'turbofan'

    # DataBase path
    aircraft_data_path = './DataBase/aircraft_test.json'
    engine_data_path = './DataBase/engine_test.json'
    mission_data_path = './Missions/fuelburn18000.json'

    data_base_args = [aircraft_data_path, engine_data_path, mission_data_path]

    # Design Variables
    dv_list = [4.7, 30.0, 1.61, 1380]

    # build design variable class
    dv = DesignVariable(propulsion_type, aircraft_type)

    # set the thermal design variables
    thermal_design_variables = dv.set_design_variable(dv_list)

    print('Thermal Design Variable:', thermal_design_variables)

    # off design variables
    off_altitude = 0.0
    off_mach = 0.0
    off_required_thrust = 133000  # [N]

    off_params_args = [off_altitude, off_mach, off_required_thrust]

    # revolve rate (0.8 ~ 1.2)
    rev_lp = 0.99
    rev_fan = 1.00
    rev_args = [rev_lp, rev_fan]

    # design point params
    design_point_params = [10668, 0.78]

    # If you want to create or set new mission, this following code have to run
    # Build Mission class
    init_mission_class = InitMission(aircraft_name, engine_name, aircraft_data_path, engine_data_path)

    # build calc design point class
    cdp = CalcDesignPoint(aircraft_name, engine_name, aircraft_type, propulsion_type, thermal_design_variables, init_mission_class, design_point_params, data_base_args)

    cdp.run_dp()

    cdp.objective_func_dp()

    # Build Calc Off design class
    cdop = CalcDesignOffPoint(aircraft_name, engine_name, aircraft_type, propulsion_type, thermal_design_variables,
                              off_params_args, design_point_params, data_base_args, cdp)

    # run off design calculation
    cdop.run_off(rev_args)

    # calculate objective function
    cdop.objective_func_doff()

    # build engine weight class
    ew = EngineWeight(aircraft_name, engine_name, aircraft_type, propulsion_type, engine_data_path, cdp, cdop)

    # run engine calculation
    ew.run_engine()

    # calculate engine amplitude
    init_mission_class.load_mission_config(mission_data_path)
    baseline_engine_weight = init_mission_class.engine_weight
    engine_amplitude = ew.total_engine_weight / baseline_engine_weight

    # engine amplitude
    aw_args = 1.0

    print(aw_args)

    # build aircraft weight class
    aw = AircraftWeight(aircraft_name, aircraft_type, ew, init_mission_class, thermal_design_variables, data_base_args, aw_args)

    aw.run_airframe()

    print(aw.weight_airframe)

def test_bwb():
    # global variables
    aircraft_name = 'A320'
    engine_name = 'V2500'
    aircraft_type = 'BWB'
    propulsion_type = 'turbofan'

    # DataBase path
    aircraft_data_path = './DataBase/aircraft_test.json'
    engine_data_path = './DataBase/engine_test.json'
    mission_data_path = './Missions/fuelburn18000.json'

    data_base_args = [aircraft_data_path, engine_data_path, mission_data_path]

    # design variables
    dv_list = [4.7, 30.0, 1.61, 1380, 1.0, 0.8, 0.48, 1.0, 0.2]

    # build design variable class
    dv = DesignVariable(propulsion_type, aircraft_type)

    # set the thermal design variables
    thermal_design_variables = dv.set_design_variable(dv_list)

    print('Thermal Design Variable:', thermal_design_variables)

    # off design variables
    off_altitude = 0.0
    off_mach = 0.0
    off_required_thrust = 133000  # [N]

    off_params_args = [off_altitude, off_mach, off_required_thrust]

    # revolve rate (0.8 ~ 1.2)
    rev_lp = 0.99
    rev_fan = 1.00
    rev_args = [rev_lp, rev_fan]

    # design point params
    design_point_params = [10668, 0.78]

    # If you want to create or set new mission, this following code have to run
    # Build Mission class
    init_mission_class = InitMission(aircraft_name, engine_name, aircraft_data_path, engine_data_path)

    # build calc design point class
    cdp = CalcDesignPoint(aircraft_name, engine_name, aircraft_type, propulsion_type, thermal_design_variables,
                          init_mission_class, design_point_params, data_base_args)

    cdp.run_dp()

    cdp.objective_func_dp()

    # set mission
    # mission_coef_args = [0.5, 0.5]
    # save_mission_data_path = './Missions/current2.json'
    # im.set_mission(mission_coef_args, save_mission_data_path)

    # Build Calc Off design class
    cdop = CalcDesignOffPoint(aircraft_name, engine_name, aircraft_type, propulsion_type, thermal_design_variables,
                              off_params_args, design_point_params, data_base_args, cdp)

    # run off design calculation
    cdop.run_off(rev_args)

    # calculate objective function
    cdop.objective_func_doff()

    # build engine weight class
    ew = EngineWeight(aircraft_name, engine_name, aircraft_type, propulsion_type, engine_data_path, cdp, cdop)

    # run engine calculation
    ew.run_engine()

    # calculate engine amplitude
    init_mission_class.load_mission_config(mission_data_path)
    baseline_engine_weight = init_mission_class.engine_weight
    engine_amplitude = ew.total_engine_weight / baseline_engine_weight

    # engine amplitude
    aw_args = 1.0

    print(aw_args)

    # build aircraft weight class
    aw = AircraftWeight(aircraft_name, aircraft_type, ew, init_mission_class, thermal_design_variables, data_base_args,
                        aw_args)

    aw.run_airframe()

    print(aw.weight_airframe)

def test_tedp():

    # global variables
    aircraft_name = 'A320'
    engine_name = 'V2500'
    aircraft_type = 'normal'
    propulsion_type = 'TeDP'

    # DataBase path
    aircraft_data_path = './DataBase/aircraft_test.json'
    engine_data_path = './DataBase/engine_test.json'
    mission_data_path = './Missions/fuelburn18000.json'

    data_base_args = [aircraft_data_path, engine_data_path, mission_data_path]

    # Design Variables
    dv_list = [30.0, 1380, 4.7, 1.61, 0.6, 0.99, 3]

    # build design variable class
    dv = DesignVariable(propulsion_type, aircraft_type)

    # set the thermal design variables
    thermal_design_variables = dv.set_design_variable(dv_list)

    print('Thermal Design Variable:', thermal_design_variables)

    # off design variables
    off_altitude = 0.0
    off_mach = 0.0
    off_required_thrust = 133000  # [N]

    off_params_args = [off_altitude, off_mach, off_required_thrust]

    # revolve rate (0.8 ~ 1.2)
    rev_lp = 1.5
    rev_fan = 1.00
    rev_args = [rev_lp, rev_fan]

    # design point params
    design_point_params = [10668, 0.78]

    # If you want to create or set new mission, this following code have to run
    # Build Mission class
    init_mission_class = InitMission(aircraft_name, engine_name, aircraft_data_path, engine_data_path)

    # build calc design point class
    cdp = CalcDesignPoint(aircraft_name, engine_name, aircraft_type, propulsion_type, thermal_design_variables,
                          init_mission_class, design_point_params, data_base_args)

    cdp.run_dp()

    cdp.objective_func_dp()

    # Build Calc Off design class
    cdop = CalcDesignOffPoint(aircraft_name, engine_name, aircraft_type, propulsion_type, thermal_design_variables,
                              off_params_args, design_point_params, data_base_args, cdp)

    # run off design calculation
    cdop.run_off(rev_args)

    # calculate objective function
    cdop.objective_func_doff()

    # build engine weight class
    ew = EngineWeight(aircraft_name, engine_name, aircraft_type, propulsion_type, engine_data_path, cdp, cdop)

    # run engine calculation
    ew.run_engine()

    # calculate engine amplitude
    init_mission_class.load_mission_config(mission_data_path)
    baseline_engine_weight = init_mission_class.engine_weight
    # Depending on the case
    # load the engine under the main wing
    engine_amplitude = (ew.core_engine_weight + ew.distributed_fan_weight) / baseline_engine_weight

    # engine amplitude
    aw_args = engine_amplitude

    print(aw_args)

    # build aircraft weight class
    aw = AircraftWeight(aircraft_name, aircraft_type, ew, init_mission_class, thermal_design_variables, data_base_args,
                        aw_args)

    aw.run_airframe()

    aw_coef = 1.1346654898593784

    print(aw.weight_airframe * aw_coef)


if __name__ == '__main__':

    # test()
    # test_bwb()
    test_tedp()





