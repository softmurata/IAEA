import numpy as np
cimport numpy as np
cimport cython
import time
from init_propulsion import InitPropulsion, InitThermalParams
ctypedef np.float64_t DOUBLE_t

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

        args = [self.qref, self.qref_e, self.dref, self.g, self.propulsion_type]
        self.fscl, self.A_core, self.A_disfan, self.core_diam, self.disfan_diam, self.airflow, self.sfc, self.isp, self.thrust_cruise = objective_func(args)




cpdef calc_design_point(str aircraft_name, str engine_name, str aircraft_type, str propulsion_type, list thermal_design_variables, init_mission_class, list design_point_params, list data_base_args, list tuning_args):
    cdef calc_design_point_class
    cdef double thrust_cruise
    cdef tuning_flag
    calc_design_point_class = CalcDesignPoint(aircraft_name, engine_name, aircraft_type, propulsion_type, thermal_design_variables, init_mission_class, design_point_params, data_base_args)
    thrust_cruise, tuning_flag = tuning_args
    if not tuning_flag:
        calc_design_point_class.dref[10] = thrust_cruise
    else:
        calc_design_point_class.dref[10] = init_mission_class.required_thrust
    calc_design_point_class.run_dp()
    calc_design_point_class.objective_func_dp()

    return calc_design_point_class

cdef objective_func(list args):
    cdef:
        np.ndarray[DOUBLE_t, ndim=2] qref
        np.ndarray[DOUBLE_t, ndim=2] qref_e
        np.ndarray[DOUBLE_t, ndim=1] dref
        double g
        str propulsion_type
        double W00
        double WE00
        double WF30
        double WF70
        double freq
        double f00
        double f19
        double f90
        double f19e
        double f00e
        double fscl
        double BPR
        double BPRe
        double thr10
        double A_core
        double A_disfan
        double core_diam
        double disdan_diam
        double TSFC
        double TISP
        double airflow
        double sfc
        double isp
        double thrust_cruise
        list results


    qref, qref_e, dref, g, propulsion_type = args
    W00 = qref[1, 0]
    WE00 = qref_e[1, 0]

    WF30 = qref[0, 30]
    WF70 = qref[0, 70]

    freq = dref[10]

    f00, f19, f90, f19e, f00e = qref[0, 0], qref[0, 19], qref[0, 90], qref_e[0, 19], \
                                    qref_e[0, 0]

    # confirmation for thrust and airflow rate
    # print('f00:', f00, 'f19:', f19, 'f90:', f90, 'f19e:', f19e, 'f00e:', f00e)
    # print('W00:', W00, 'WE00:', WE00, 'WF30:', WF30)
    # print('required thrust:', freq)
    fscl = freq / (f00 + f19 + f90 + f19e + f00e)

    # print('fscl:',fscl)

    BPR = dref[20]
    BPRe = dref[31]

    thr10 = 0.32

    # Area of front core
    A_core = 0

    if propulsion_type in ['turbofan', 'PartialElectric']:
        A_core = qref[2, 10]
    elif propulsion_type in ['turbojet', 'turboshaft', 'TeDP']:
        A_core = qref[2, 20]
    # Area of distributed fan
    A_disfan = qref_e[2, 10]
    # Diameter of front core
    core_diam = 2.0 * np.sqrt(fscl * A_core / (1.0 - thr10 ** 2) / np.pi)
    # Diameter of distributed fan
    disfan_diam = 2.0 * np.sqrt(fscl * A_disfan / (1.0 - thr10 ** 2) / np.pi)

    # description of results at design point
    print('')
    print('=' * 5 + ' DESIGN POINT RESULTS ' + '=' * 5)
    # print('core diameter:', self.core_diam, 'disfan diameter:', self.disfan_diam)

    TSFC = (WF30 + WF70) * g * 3600 / (f00 + f19 + f90 + f19e + f00e)
    TISP = ((f00 + f19 + f90 + f19e + f00e) / g) / (W00 + WE00)

    print('SFC:', TSFC)
    print('ISP:', TISP)
    print('BPRe:', BPRe)

    airflow = TISP * (W00 + WE00)
    print('airflow:', airflow)
    print('=' * 40)
    print('')

    # Change according to description
    sfc = TSFC
    isp = TISP

    thrust_cruise = freq

    results = [fscl, A_core, A_disfan, core_diam, disfan_diam, airflow, sfc, isp, thrust_cruise]

    return results



