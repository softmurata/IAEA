import numpy as np
from AirComponent import InitAirParams
from air_shape import *
from init_airparams import InitAirParams
from standard_air_utils import StandardAir

# for test
from mission_tuning import InitMission
from total_mission_tuning import TotalMission
from design_variable import DesignVariable
from thermal_dp import CalcDesignPoint
from thermal_doff import CalcDesignOffPoint
from engine_weight import EngineWeight


# calculate Lift and Drag
class FluidDynamic(object):
    """

    """

    def __init__(self, aircraft_name, init_mission_class, air_shape_class, engine_weight_class, flight_states, other_args):
        self.aircraft_name = aircraft_name
        # engine_amplitude includes other args
        engine_amplitude = other_args
        # build air shape class (Already built)
        self.air_shape_class = air_shape_class
        # build engine weight class (Already built)
        self.engine_weight_class = engine_weight_class

        flight_mach, flight_altitude, flight_angle = flight_states
        # current_mach
        self.current_mach = flight_mach
        self.current_altitude = flight_altitude
        self.attack_of_angle = flight_angle * np.pi / 180.0
        self.zero_attack_of_angle = 0  # Now assume NACA63

        # standard air class
        self.sa = StandardAir(self.current_altitude)

        # cruise state
        self.current_state = 'subsonic' if self.current_mach < 1.0 else 'supersonic'
        # beta
        self.beta = np.sqrt(1.0 - self.current_mach ** 2) if self.current_mach < 1.0 else np.sqrt(self.current_mach ** 2 - 1)

        # build init air shape class
        self.init_airshape_class = InitAirParams(self.aircraft_name, init_mission_class, engine_amplitude)
        self.init_airshape_class.set_main_config()

        # Lift by Drag
        self.l_d_max = None
        self.l_d_cruise = None

        # slope of lift
        self.slope_of_lift = None  # [rad]
        # total shape drag
        self.cd0 = 0
        self.cd = None  # coefficient of drag
        self.cl = None  # coefficient of lift

    def calc_lift_by_drag_ratio(self, calc_type='static'):
        """

        :param calc_type: the method of calculating the values of lift by drag
        :return: None
        """

        if calc_type == 'static':
            self.static_lift_by_drag()

        else:
            # calculate lift coefficient
            self.calc_lift()
            # calculate drag coefficient
            self.calc_drag()
            self.l_d_cruise = self.cl / self.cd

    # statistically calculate Lift by Drag
    def static_lift_by_drag(self):
        """
        calculate lift by drag

        """
        # calculate wet aspect ratio
        main_wing_wet_area = 2.0 * self.air_shape_class.main_wing_wet_area * (1.0 + 0.25 * self.init_airshape_class.tcroot * (1.0 + 0.7 * self.init_airshape_class.Lambda) / (1.0 + self.init_airshape_class.Lambda))
        if self.air_shape_class.name == 'normal':
            hori_wing_wet_area = 2.0 * self.air_shape_class.hori_wing_wet_area * (1.0 + 0.25 * self.init_airshape_class.tcroot * (1.0 + 0.7 * self.init_airshape_class.Lambda) / (1.0 + self.init_airshape_class.Lambda))
            vert_wing_wet_area = 2.0 * self.air_shape_class.vert_wing_wet_area * (1.0 + 0.25 * self.init_airshape_class.tcroot * (1.0 + 0.7 * self.init_airshape_class.Lambda) / (1.0 + self.init_airshape_class.Lambda))
        elif self.air_shape_class.name == 'BWB':
            hori_wing_wet_area = 0
            vert_wing_wet_area = 0

        fuselage_wet_area = self.air_shape_class.fuselage_wet_area
        # fuselage_wet_area = self.air_shape_class.fuselage_surface_area

        total_wet_area = main_wing_wet_area + hori_wing_wet_area + vert_wing_wet_area + fuselage_wet_area

        # total_wet_area = self.air_shape_class.wet_area

        # print(fuselage_wet_area, main_wing_wet_area, hori_wing_wet_area, vert_wing_wet_area)

        if self.air_shape_class.name == 'normal':
            AR = self.init_airshape_class.AR
            Swref = self.init_airshape_class.Swref
        elif self.air_shape_class.name == 'BWB':
            AR = self.air_shape_class.bwb_AR
            Swref = self.air_shape_class.bwb_Sw

        wet_ar = AR / (total_wet_area / Swref)
        # calculate lift by drag statistically
        self.l_d_max = 14.6 * np.sqrt(wet_ar)
        self.l_d_cruise = 0.866 * self.l_d_max
        print('')
        print('=' * 10 + 'Aero Dynamic perfomances' + '=' * 10 )
        print('Total wet area:', total_wet_area)
        print('Lift by Drag Max:', self.l_d_max, 'Lift by Drag cruise:', self.l_d_cruise, 'wet ar:', wet_ar)
        print('=' * 40)
        print('')

    def make_lift_curve_slope(self):
        """
        statistically calculating slope of lift curve for acquiring the value of lift coefficient

        """
        # Initialize fuselage diameter and main wing width
        fuselage_d = 0  # [ft]
        wing_b = 0  # [ft]
        S_exposed = 0  # [ft**2]

        if self.air_shape_class.name == 'normal':
            fuselage_d = self.air_shape_class.df
            wing_b = self.air_shape_class.BW
            S_exposed = self.init_airshape_class.Sw
            AR = self.init_airshape_class.AR
            Swref = self.init_airshape_class.Swref
        elif self.air_shape_class.name == 'BWB':
            fuselage_d = (self.air_shape_class.section_lengths[1][1] + self.air_shape_class.t1) * 2
            wing_b = self.air_shape_class.bwb_BW * 2
            S_exposed = self.init_airshape_class.Swref
            AR = self.air_shape_class.bwb_AR
            Swref = self.air_shape_class.bwb_Sw

        # print('fuselage d:', fuselage_d)
        # print('wing b:', wing_b)
        # print('AR:', AR)
        # print('Sw:', S_exposed, 'Swref:', Swref)

        # helper coefficients for calculating slope of lift
        F = 1.07 * (1.0 + (fuselage_d / wing_b)) ** 2
        eta = 0.95

        if 0 <= self.current_mach <= 0.9:
            self.slope_of_lift = (2 * np.pi * AR) / (2.0 + np.sqrt(4.0 + AR ** 2 * self.beta ** 2 / eta ** 2 * (1.0 + np.tan(self.init_airshape_class.theta * np.pi / 180.0) ** 2 / self.beta ** 2))) * (S_exposed / Swref) * F

        elif 1.2 <= self.current_mach:
            self.slope_of_lift = 4.0 / self.beta

        else:
            slope_of_lift_left = 9.158680497060704
            slope_of_lift_right = 6.030226891555273
            a_coef = -(slope_of_lift_right - slope_of_lift_left) / (0.2 ** 2 - 0.1 ** 2)
            b_coef = slope_of_lift_left + 0.1 ** 2 * a_coef
            self.slope_of_lift = -a_coef * (self.current_mach - 1) ** 2 + b_coef

    # calculate lift coefficient
    def calc_lift(self):
        self.make_lift_curve_slope()
        self.cl = self.slope_of_lift * (self.attack_of_angle - self.zero_attack_of_angle)
        print('total lift coefficient:', self.cl)

    # calculate drag coefficients
    # shape drag, misc drag, L&P drag, wave drag(if supersonic)
    def shape_drag(self, component_name, representative_length, d, surface_state='smooth paint'):
        """

        :param component_name:
        :param representative_length:
        :param d: representative dismater
        :param surface_state: the name of surface state
        :return: component_friction_coef
        """
        # calculate Reynolds number
        Re = self.sa.rou * self.current_mach * self.sa.a * representative_length / self.sa.mu
        # define surface roughness
        surface_rough = 1.0e-5
        if surface_state == 'smooth paint':
            surface_rough = 2.08 * 1e-5  # [ft] 0.634 * 1e-5 if [m]
        elif surface_rough == 'polished sheet metal':
            surface_rough = 0.50 * 1e-5
        elif surface_state == 'camouflage paint':
            surface_rough = 3.33 * 1e-5
        elif surface_state == 'production sheet metal':
            surface_rough = 1.33 * 1e-5
        elif surface_state == 'smooth molded composite':
            surface_rough = 0.17 * 1e-5

        # helper function
        # friction coefficient
        def calc_fric_coef(mach, Reynolds_number, flow_state):
            Cf = 0
            if flow_state == 'laminar':
                Cf = 1.328 / np.sqrt(Reynolds_number)
            elif flow_state == 'turbulence':
                Cf = 0.455 / ((np.log10(Reynolds_number)) ** 2.58 * (1.0 + 0.144 * mach ** 2) ** 0.65)

            return Cf

        # Initialize friction coefficients
        friction_coef = 0

        if self.current_state == 'subsonic':
            Re_cutoff = 38.2 * (representative_length / surface_rough) ** 1.053
            if Re < Re_cutoff:
                flow_state = 'laminar'
            else:
                flow_state = 'turbulence'
            friction_coef = calc_fric_coef(self.current_mach, Re, flow_state)

        elif self.current_state == 'supersonic':
            Re_cutoff = 44.62 * (representative_length / surface_rough) ** 1.053 * self.current_mach ** 1.16
            if Re < Re_cutoff:
                flow_state = 'laminar'
            else:
                flow_state = 'turbulence'

            friction_coef = calc_fric_coef(self.current_mach, Re, flow_state)

        # representative length by diameter
        f = representative_length / d

        FF = 0

        if component_name in ['main wing', 'horizontal wing', 'vertical wing', 'strut', 'pylon']:
            FF = (1.0 + (0.6 / 0.5) * self.init_airshape_class.tcroot + 100 * self.init_airshape_class.tcroot ** 4) * (1.34 * self.current_mach ** 0.18 * (np.cos(self.init_airshape_class.theta * np.pi / 180.0)) ** 0.28)
        elif component_name in ['fuselage', 'canopy']:
            FF = (1.0 + 60 / f ** 2 + f / 400)
        elif component_name in ['Nacelle']:
            FF = 1.0 + 0.35 / f

        if self.air_shape_class.name == 'BWB' and component_name == 'fuselage':
            df = (self.air_shape_class.section_lengths[1][1] + self.air_shape_class.t1) * 2
            tcroot = self.air_shape_class.cabin_height / (df * 2) * 0.5
            FF = (1.0 + (0.6 / 0.5) * tcroot + 100 * tcroot ** 4)

        Q = 1.0

        if component_name in ['Nacelle']:
            Q = 1.5
        elif component_name in ['main wing', 'horizontal wing', 'vertical wing', 'fuselage']:
            Q = 1.0

        # calculate component friction coefficients
        component_friction_coef = friction_coef * FF * Q

        return component_friction_coef

    def misc_drag(self):
        # aft fuselage drag
        if self.air_shape_class.name == 'BWB':
            theta_back = 5
            df = (self.air_shape_class.section_lengths[2][0] + self.air_shape_class.t2) * 2
            amax = (df) ** 2 * np.pi / 4
        elif self.air_shape_class.name == 'normal':
            theta_back = self.air_shape_class.theta_back
            amax = self.air_shape_class.df ** 2 * np.pi / 4

        u = theta_back * np.pi / 180
        cdrag_misc = 3.83 * u ** 2.5 * amax

        # landing gear effect
        landing_gear_effect_coef = 1.07
        if self.current_altitude <= 0.0:
            cdrag_misc *= landing_gear_effect_coef

        return cdrag_misc

    def lp_drag(self):
        # ratio of shape drag is 3% ~ 5%
        return self.cd0 * 0.05

    def wave_drag(self, Sref):
        """
        calculate wave drag

        :param Sref: the reference value of main wing
        :return: cdrag_wave
        """
        # calculate sheer-hack drag
        if self.air_shape_class.name == 'normal':
            amax = self.air_shape_class.df ** 2 * np.pi / 4
            longtitude = self.air_shape_class.lf
        elif self.air_shape_class.name == 'BWB':
            df = (self.air_shape_class.section_lengths[1][1] + self.air_shape_class.t1) * 2
            amax = (df) ** 2 * np.pi / 4
            longtitude = self.air_shape_class.overall_chord
        sheer_hack_drag = 4.5 * np.pi * (amax / longtitude) ** 2

        # calculate parasite drag
        cdrag_wave = 0
        if self.current_state == 'supersonic':
            cdrag_wave = 1.0 * (1.0 - 0.386 * (self.current_mach - 1.2) ** 0.57 * (1.0 - self.init_airshape_class.theta ** 0.77 / 100)) * sheer_hack_drag
        elif self.current_state == 'subsonic':
            cdrag_wave = 0

        cdrag_wave /= Sref

        MDD = 0.8 * 0.9 - 0.05 * 0.4  # MDD = MDD(L=0) * LFDD - 0.05 * CL_design
        Mcr = 0.8
        if Mcr <= self.current_mach <= 1.05:
            cdrag_wave = 0.002 / (MDD - Mcr) ** 2 * (self.current_mach - Mcr) ** 2

        return cdrag_wave

    def calc_drag(self):
        """
        main function

        calculate total drag

        :return: None
        """
        # Reference Main wing area
        Sref = self.init_airshape_class.Swref
        # Main Wing
        component_name = 'main wing'
        Swet = self.air_shape_class.main_wing_wet_area
        representative_length = 0.5 * (self.air_shape_class.main_croot + self.air_shape_class.main_ctip)
        d = 1  # meaningless

        cffq_mw = self.shape_drag(component_name, representative_length, d)
        # calculate cdrag
        cdrag_mw = cffq_mw * Swet / Sref

        # Vertical Wing
        component_name = 'vertical wing'
        Swet = self.air_shape_class.vert_wing_wet_area
        representative_length = 0.5 * (self.air_shape_class.vert_croot + self.air_shape_class.vert_ctip)
        if Swet is None:
            Swet = 0
            representative_length = 0
        d = 1  # meaningless

        cffq_vw = self.shape_drag(component_name, representative_length, d)

        # calculate cdrag
        cdrag_vw = cffq_vw * Swet / Sref

        # Horizontal Wing
        component_name = 'horizontal wing'
        Swet = self.air_shape_class.hori_wing_wet_area
        representative_length = 0.5 * (self.air_shape_class.hori_croot + self.air_shape_class.hori_ctip)
        if Swet is None:
            Swet = 0
            representative_length = 0

        d = 1  # meaningless

        cffq_hw = self.shape_drag(component_name, representative_length, d)

        # calculate cdrag
        cdrag_hw = cffq_hw * Swet / Sref

        # Fuselage
        component_name = 'fuselage'
        Swet = self.air_shape_class.fuselage_wet_area
        representative_length = 0
        d = 1
        if self.air_shape_class.name == 'normal':
            representative_length = self.air_shape_class.fuselage_length
            d = self.air_shape_class.df
        elif self.air_shape_class.name == 'BWB':
            representative_length = self.air_shape_class.overall_chord
            d = (self.air_shape_class.section_lengths[1][1] + self.air_shape_class.t1) * 2

        print('fuselage')
        print(representative_length, d)
        cffq_fus = self.shape_drag(component_name, representative_length, d)

        # calculate cdrag
        cdrag_fus = cffq_fus * Swet / Sref

        # Nacelle
        component_name = 'Nacelle'
        representative_length = self.engine_weight_class.total_engine_length
        d = self.engine_weight_class.inlet_diameter[0, 10]  # fan diameter
        if d <= 0.0:
            d = self.engine_weight_class.inlet_diameter[0, 20]  # Core diameter
        Swet = np.pi * (0.5 * d) ** 2 * representative_length

        cffq_nac = self.shape_drag(component_name, representative_length, d)

        # calculate cdrag
        cdrag_nac = cffq_nac * Swet / Sref

        # misc drag
        cdrag_misc = self.misc_drag() / Sref

        # wave drag
        cdrag_wave = self.wave_drag(Sref)

        self.cd0 = cdrag_mw + cdrag_hw + cdrag_vw + cdrag_fus + cdrag_nac + cdrag_misc + cdrag_wave

        # leakage and passage drag
        cdrag_lp = self.lp_drag()

        self.cd0 += cdrag_lp

        print('main wing:', cdrag_mw, 'vertical wing:', cdrag_vw, 'horizontal wing:', cdrag_hw)
        print('fuselage:', cdrag_fus, 'nacelle:', cdrag_nac)
        print('misc:', cdrag_misc, 'L&P:', cdrag_lp, 'wave:', cdrag_wave)
        print('cd0:', self.cd0)

        if self.air_shape_class.name == 'normal':
            AR = self.init_airshape_class.AR
        elif self.air_shape_class.name == 'BWB':
            AR = self.air_shape_class.bwb_AR

        # induced drag
        e = 1.78 * (1 - 0.045 * AR ** 0.68) - 0.64
        print('straight e:', e)

        if self.init_airshape_class.theta > 30:
            e = 4.61 * (1.0 - 0.045 * AR ** 0.68) * (np.cos(self.init_airshape_class.theta)) ** 0.15 - 3.1

        K = 1 / (np.pi * AR * e)

        self.cd = self.cd0 + K * self.cl ** 2

        print('total drag coefficient:', self.cd)


# test code
# Normal shape aircraft
def test_normal_shape():

    aircraft_name = 'A320'
    engine_name = 'V2500'
    aircraft_type = 'normal'
    propulsion_type = 'turbofan'

    # data base path
    aircraft_data_path = './DataBase/aircraft_test.json'
    engine_data_path = './DataBase/engine_test.json'
    mission_data_path = './Missions/fuelburn18000.json'

    data_base_args = [aircraft_data_path, engine_data_path, mission_data_path]

    # build init mission class
    init_mission_class = InitMission(aircraft_name, engine_name, aircraft_data_path, engine_data_path)
    # load mission data which has already tuned
    init_mission_class.load_mission_config(mission_data_path)

    # build design variable class
    dv = DesignVariable(propulsion_type, aircraft_type)
    dv_list = [4.7, 30.0, 1.61, 1380, 0, 0, 0, 0, 0]

    # determine thermal design variables
    thermal_design_variables = dv.set_design_variable(dv_list)

    print('thermal design variables:', thermal_design_variables)

    # set off design point params
    off_altitude = 0
    off_mach = 0
    off_required_thrust = 133000
    off_param_args = [off_altitude, off_mach, off_required_thrust]

    # build off design point class
    calc_off_design_point_class = CalcDesignOffPoint(aircraft_name, engine_name, aircraft_type, propulsion_type, thermal_design_variables, off_param_args, data_base_args)
    # run off design point's calculation
    rev_args = [0.98, 1.0]  # [lp shaft param, distributed fan params]
    calc_off_design_point_class.run_off(rev_args)
    # calculate objective indexes
    calc_off_design_point_class.objective_func_doff()

    # build engine class
    ew = EngineWeight(aircraft_name, engine_name, aircraft_type, propulsion_type, calc_off_design_point_class, engine_data_path)
    # calculate engine weight and params
    ew.run_engine()

    # build total mission class
    baseline_aircraft_name = 'A320'
    baseline_aircraft_type = 'normal'
    baseline_engine_name = 'V2500'
    baseline_propulsion_type = 'turbofan'
    baseline_aircraft_data_path = './DataBase/aircraft_test.json'
    baseline_engine_data_path = './DataBase/engine_test.json'

    baseline_args = [(baseline_aircraft_name, baseline_aircraft_type), (baseline_engine_name, baseline_propulsion_type), (baseline_aircraft_data_path, baseline_engine_data_path)]

    tm = TotalMission(baseline_args, off_param_args, mission_data_path)

    tm.load_coef_config()

    # calculate engine amplitude
    engine_amplitude = ew.total_engine_weight * tm.engine_weight_coef / init_mission_class.engine_weight

    # build air shape class
    engine_mounting_positions = [0.2, 0.2]
    ns = NormalShape(aircraft_name, init_mission_class, ew, engine_mounting_positions, engine_amplitude)

    ns.run_airshape(aircraft_data_path)

    # set flight state
    flight_state = [0.78, 10668, 4.2]

    # Build Fluid dynamic class
    fd = FluidDynamic(aircraft_name, init_mission_class, ns, ew, flight_state, engine_amplitude)

    # calculate lift coefficient
    fd.calc_lift()

    # calculate drag coefficient
    fd.calc_drag()

    # calculate lift by drag
    fd.calc_lift_by_drag_ratio()

    l_d_cruise = fd.cl / fd.cd
    print('')
    print('L/D:', l_d_cruise)

# test code
# Blended wing body
def test_bwb():
    aircraft_name = 'A320'
    engine_name = 'V2500'
    aircraft_type = 'BWB'
    propulsion_type = 'TeDP'

    # data base path
    aircraft_data_path = './DataBase/aircraft_test.json'
    engine_data_path = './DataBase/engine_test.json'
    mission_data_path = './Missions/fuelburn18000.json'

    data_base_args = [aircraft_data_path, engine_data_path, mission_data_path]

    # build init mission class
    init_mission_class = InitMission(aircraft_name, engine_name, aircraft_data_path, engine_data_path)
    # load mission data which has already tuned
    init_mission_class.load_mission_config(mission_data_path)

    # build design variable class
    dv = DesignVariable(propulsion_type, aircraft_type)
    # equal to the A320 bottom area
    dv_list = [40.0, 1430, 5.0, 1.24, 0.15, 0.99, 3, 1.0, 0.8, 0.48, 1.0, 0.2]

    # [0.4, 0.5, 0.1] => [0.83, 0.7, 0.28, 0.1] => 17.64
    # [0.6, 0.3, 0.1] => [0.35, 0.31, 0.18, 0.1] => 18.67

    # determine thermal design variables
    thermal_design_variables = dv.set_design_variable(dv_list)

    print('thermal design variables:', thermal_design_variables)

    # set off design point params
    off_altitude = 0
    off_mach = 0
    off_required_thrust = 133000
    off_param_args = [off_altitude, off_mach, off_required_thrust]

    # design point parameters
    design_point_params = [10668, 0.78]

    # build calc design point class
    calc_design_point_class = CalcDesignPoint(aircraft_name, engine_name, aircraft_type, propulsion_type, thermal_design_variables, init_mission_class, design_point_params, data_base_args)

    calc_design_point_class.run_dp()

    calc_design_point_class.objective_func_dp()

    # build off design point class
    calc_off_design_point_class = CalcDesignOffPoint(aircraft_name, engine_name, aircraft_type, propulsion_type,
                                                     thermal_design_variables, off_param_args, design_point_params, data_base_args, calc_design_point_class)
    # run off design point's calculation
    rev_args = [1.391, 1.0]  # [lp shaft param, distributed fan params]
    calc_off_design_point_class.run_off(rev_args)
    # calculate objective indexes
    calc_off_design_point_class.objective_func_doff()

    # build engine class
    ew = EngineWeight(aircraft_name, engine_name, aircraft_type, propulsion_type, engine_data_path, calc_design_point_class, calc_off_design_point_class)
    # calculate engine weight and params
    ew.run_engine()

    # build total mission class
    baseline_aircraft_name = 'A320'
    baseline_aircraft_type = 'normal'
    baseline_engine_name = 'V2500'
    baseline_propulsion_type = 'turbofan'
    baseline_aircraft_data_path = './DataBase/aircraft_test.json'
    baseline_engine_data_path = './DataBase/engine_test.json'

    baseline_args = [(baseline_aircraft_name, baseline_aircraft_type), (baseline_engine_name, baseline_propulsion_type),
                     (baseline_aircraft_data_path, baseline_engine_data_path)]

    design_point = ['cruise', design_point_params]

    tm = TotalMission(baseline_args, design_point, off_param_args, mission_data_path)

    tm.load_coef_config()

    # calculate engine amplitude
    engine_amplitude = ew.total_engine_weight * tm.engine_weight_coef / init_mission_class.engine_weight

    # build air shape class
    engine_mounting_positions = [0.9, 0.9]
    cabin_ratio = 0.8
    init_shape_params = cabin_ratio
    bwbs = BlendedWingBodyShape(aircraft_name, init_mission_class, thermal_design_variables, ew, init_shape_params, data_base_args, engine_amplitude)
    bwbs.run_airshape()

    # set flight state
    flight_state = [0.78, 10668, 4.2]

    # Build Fluid dynamic class
    fd = FluidDynamic(aircraft_name, init_mission_class, bwbs, ew, flight_state, engine_amplitude)

    # calculate lift coefficient
    fd.calc_lift()

    # calculate drag coefficient
    fd.calc_drag()

    # calc lift by drag
    fd.calc_lift_by_drag_ratio()

    l_d_cruise = fd.cl / fd.cd
    print('')
    print('L/D:', l_d_cruise)



if __name__ == '__main__':
    # test_normal_shape()

    test_bwb()


