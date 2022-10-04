import numpy as np
import time
from EngineComponent import *
from init_propulsion import InitPropulsion, InitThermalParams
from mission_tuning import InitMission
from design_variable import DesignVariable

"""
もしかしたら、バッテリーを用いる推進機構と用いない推進機構では計算の仕方に違いが生じると考えられる
1. Battery
  0. Determine baseline aircraft type and engine type
  1. calculate design point and off design point at baseline systems
  2. get max entarpy(J) from the baseline system's results
  3. set the distributed ratio of entarpy between baseline system and battery system
"""


class CalcDesignPoint(InitThermalParams):
    """
    Note:
        Computational Algorithm:

    """

    def __init__(self, aircraft_name, engine_name, aircraft_type, propulsion_type, thermal_design_variables, init_mission_class, design_point_params,
                 data_base_args):

        # set file path including the configurations
        aircraft_data_path, engine_data_path, mission_data_path = data_base_args

        # Succeed InitThermalParams
        super().__init__(init_mission_class, design_point_params, mission_data_path)

        self.aircraft_name = aircraft_name
        self.engine_name = engine_name
        self.aircraft_type = aircraft_type
        self.propulsion_type = propulsion_type

        # design variables
        self.thermal_design_variables = thermal_design_variables

        # set cref, dref, cref_e
        # cref => ndarray containing the elements of core compressor efficiency
        # dref => ndarray including the design variables
        # cref_e => ndarray including the element of distributed fan's efficiency
        self.build(self.thermal_design_variables)

        # component class objects are established
        self.build_component_class_object()

        # set result array
        self.qref = np.zeros((100, 100))
        self.qref_e = np.zeros((100, 100))
        # Gravity
        self.g = 9.81

        self.electric_component_density = self.init_mission_class.electric_component_density

    def build_component_class_object(self):

        # Build Init Proplusion class
        ip = InitPropulsion()

        # required component class objects
        component_classes = ip.get_component_classes(self.propulsion_type)

        # build class object of components
        self.build_component_classes_core = []
        self.build_component_classes_elec = []
        # divide core components and distributed fan components
        self.core_classes, self.dist_classes = component_classes

        for e_class in self.dist_classes:
            self.build_component_classes_elec.append(e_class(self.cref_e, self.dref, self.propulsion_type))

        for c_class in self.core_classes:
            self.build_component_classes_core.append(c_class(self.cref, self.dref, self.propulsion_type))

    # print(self.build_component_classes_core)
    # print(self.build_component_classes_elec)

    def run_dp(self):
        # Only Jet Engine
        if self.propulsion_type in ['turbojet', 'turboshaft', 'turbofan']:
            self.qref = self.run_core()

        # Jet Engine + Electric
        if self.propulsion_type in ['TeDP', 'PartialElectric']:
            # run electric part convergence
            self.qref_e = self.run_electric()

        # Jet Engine + Battery
        if self.propulsion_type in ['hybridturbojet', 'hybridturbofan']:

            self.run_battery()


    def run_core(self):

        for b_class in self.build_component_classes_core:
            self.qref = b_class(self.qref, self.qref_e)

        return self.qref

    def run_battery(self):

        pass

    def run_electric(self):

        for e_class in self.build_component_classes_elec:
            self.qref_e = e_class(self.qref_e)

        for c_class in self.build_component_classes_core:
            self.qref = c_class(self.qref, self.qref_e)

        return self.qref_e

    # acquire sfc, isp and so on
    def objective_func_dp(self):

        W00 = self.qref[1, 0]
        WE00 = self.qref_e[1, 0]

        WF30 = self.qref[0, 30]
        WF70 = self.qref[0, 70]

        freq = self.dref[10]

        f00, f19, f90, f19e, f00e = self.qref[0, 0], self.qref[0, 19], self.qref[0, 90], self.qref_e[0, 19], \
                                    self.qref_e[0, 0]

        # confirmation for thrust and airflow rate
        # print('f00:', f00, 'f19:', f19, 'f90:', f90, 'f19e:', f19e, 'f00e:', f00e)
        # print('W00:', W00, 'WE00:', WE00, 'WF30:', WF30)
        # print('required thrust:', freq)
        self.fscl = freq / (f00 + f19 + f90 + f19e + f00e)

        # print('fscl:',fscl)

        BPR = self.dref[20]
        BPRe = self.dref[31]

        thr10 = 0.32

        # Area of front core
        self.A_core = 0

        if self.propulsion_type in ['turbofan', 'PartialElectric']:
            self.A_core = self.qref[2, 10]
        elif self.propulsion_type in ['turbojet', 'turboshaft', 'TeDP']:
            self.A_core = self.qref[2, 20]
        # Area of distributed fan
        self.A_disfan = self.qref_e[2, 10]
        # Diameter of front core
        self.core_diam = 2.0 * np.sqrt(self.fscl * self.A_core / (1.0 - thr10 ** 2) / np.pi)
        # Diameter of distributed fan
        self.disfan_diam = 2.0 * np.sqrt(self.fscl * self.A_disfan / (1.0 - thr10 ** 2) / np.pi)

        # description of results at design point
        print('')
        print('=' * 5 + ' DESIGN POINT RESULTS ' + '=' * 5)
        # print('core diameter:', self.core_diam, 'disfan diameter:', self.disfan_diam)

        TSFC = (WF30 + WF70) * self.g * 3600 / (f00 + f19 + f90 + f19e + f00e)
        TISP = ((f00 + f19 + f90 + f19e + f00e) / self.g) / (W00 + WE00)

        print('SFC:', TSFC)
        print('ISP:', TISP)
        print('BPRe:', BPRe)

        self.airflow = TISP * (W00 + WE00)
        print('airflow:', self.airflow)
        print('=' * 40)
        print('')

        # Change according to description
        self.sfc = TSFC
        self.isp = TISP

        self.thrust_cruise = freq


def test():
    # turbofan test
    aircraft_name = 'A320'
    engine_name = 'V2500'
    aircraft_type = 'normal'
    propulsion_type = 'TeDP'
    # propulsion_type = 'PartialElectric'
    # propulsion_type = 'turbofan'

    dv_list = [30.0, 1380, 5.0, 1.6, 0.2, 0.99, 3]  # TeDP
    # dv_list = [3.7,30.0,1.7,1380,5.0,1.5,0.9,0.99,3]#PartialElectric
    # dv_list = [4.7, 30.0, 1.61, 1380]

    # build design variable class
    dv = DesignVariable(propulsion_type, aircraft_type)

    thermal_design_variables = dv.set_design_variable(dv_list)

    print(thermal_design_variables)

    # database args
    aircraft_data_path = './DataBase/aircraft_test.json'
    engine_data_path = './DataBase/engine_test.json'
    mission_data_path = './Missions/fuelburn18000.json'

    data_base_args = [aircraft_data_path, engine_data_path, mission_data_path]

    # design point parameters
    design_point_params = [10668, 0.78]

    # build initial mission class
    init_mission_class = InitMission(aircraft_name, engine_name, aircraft_data_path, engine_data_path)

    # build calculation design point
    cdp = CalcDesignPoint(aircraft_name, engine_name, aircraft_type, propulsion_type, thermal_design_variables, init_mission_class, design_point_params,
                          data_base_args)

    # running
    cdp.run_dp()

    # draw results
    cdp.objective_func_dp()

    # thrust cruise
    print(cdp.thrust_cruise)

    """
	#turbofan test
	aircraft_name = 'A320'
	engine_name = 'V2500'
	aircraft_type = 'normal'
	propulsion_type = 'turbofan'

	dv_list = [4.7,30.0,1.66,1380]

	#build design variable class
	dv = DesignVariable(propulsion_type, aircraft_type)

	thermal_design_variables = dv.set_design_variable(dv_list)

	print(thermal_design_variables)
	
	#build calculation design point
	cdp = CalcDesignPoint(aircraft_name, engine_name, aircraft_type, propulsion_type, thermal_design_variables)

	#running
	cdp.run()

	#draw results
	cdp.objective_func()
	"""


if __name__ == '__main__':
    start = time.time()
    test()
    finish = time.time()
    print(finish - start)
