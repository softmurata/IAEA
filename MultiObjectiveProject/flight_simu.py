import numpy as np
from aircraft_weight import AircraftWeight
from fluid_dynamic import FluidDynamic

# This class is to set the flight state and calculate lift by drag
# Also, by using such results, conduct sizing

class FlightSimulation(object):

    def __init__(self, aircraft_name, aircraft_type, init_mission_class, total_mission_class, data_base_args, ld_calc_type):

        self.aircraft_name = aircraft_name
        self.aircraft_type = aircraft_type
        # Define required class object
        self.init_mission_class = init_mission_class
        self.total_mission_class = total_mission_class

        # set the data path (aircraft data path, engine data path, mission data path)
        self.data_base_args = data_base_args

        # set the lift and drag calculation type (constant-static, constant-detail)
        self.ld_form, self.ld_calc_type = ld_calc_type.split('-')
        # flight states [[altitude
        self.flight_state_names = ['warmup', 'takeoff', 'up', 'cruise', 'down', 'approach']
        self.flight_states = []

        # set flight states
        self.set_flight_states()

        # baseline indexes
        self.baseline_max_takeoff_weight = self.init_mission_class.max_takeoff_weight
        self.baseline_aircraft_weight = self.init_mission_class.aircraft_weight
        self.baseline_required_thrust_ground = self.init_mission_class.required_thrust_ground
        self.baseline_engine_weight = self.init_mission_class.engine_weight
        self.baseline_engine_num = self.init_mission_class.engine_num

        # aircraft weight ratio (default) # previous list of constant:[0.995, 0.99, 0.98, 0.0, 0.990, 0.990]
        self.aircraft_weight_ratio = [0.995, 0.99, 0.98379, 0.0, 0.990, 0.995]

        # loiter time [h]
        self.loiter_time = 0  # 0 => no loiter

        # sfc fix coefficient
        self.sfc_coef = 1.0

        # control type => engine, engine_add, aircraft+engine, aircraft+engine+electric, max_takeoff_weight
        self.control_type = 'engine_add'
        self.engine_alpha = 0.1  # control coefficient for state of 'engine_add'
        self.baseline_volume_eff = 1.0
        self.volume_eff = 1.0

    # set the flight states [mach, altitude, attack of angle, calc_bool]
    def set_flight_states(self):

        for state_name in self.flight_state_names:
            target = None
            if state_name == 'warmup':
                target = [0, 0, 0, False]
            elif state_name == 'takeoff':
                target = [0.24, 0, 5, False]
            elif state_name == 'up':
                target = [(0.24 + self.init_mission_class.mach) * 0.5, (0 + self.init_mission_class.altitude) * 0.5, 6, False]

            elif state_name == 'cruise':
                target = [self.init_mission_class.mach, self.init_mission_class.altitude, 4, True]

            elif state_name == 'down':
                target = [(0.3 + self.init_mission_class.mach) * 0.5, (0 + self.init_mission_class.altitude) * 0.5, 6, False]

            elif state_name == 'approach':
                target = [0.3, 0, 4, False]

            self.flight_states.append(target)

    # Weight Model Part
    def control_sizing_amplitude(self, weight_results):
        engine_weight, electric_weight, aircraft_weight, max_takeoff_weight = weight_results
        sizing_amplitude = 1.0

        if self.control_type == 'engine':
            # engine amplitude
            sizing_amplitude = engine_weight / self.baseline_engine_weight

        elif self.control_type == 'engine_add':
            sizing_amplitude = 1.0

        elif self.control_type == 'aircraft+engine':
            # print('volume effect efficient:', self.volume_eff, 'baseline:', self.baseline_volume_eff)
            self.volume_eff = 1.0
            self.baseline_volume_eff = self.volume_eff
            target_weight = aircraft_weight * self.volume_eff + engine_weight * self.baseline_engine_num
            baseline_target_weight = self.baseline_aircraft_weight * self.baseline_volume_eff + self.baseline_engine_weight * self.baseline_engine_num
            sizing_amplitude = target_weight / baseline_target_weight

        elif self.control_type == 'aircraft+engine+electric':
            pass

        elif self.control_type == 'max_takeoff_weight':
            pass

        # print('')
        # print('sizing amplitude:', sizing_amplitude)
        # print('')

        return sizing_amplitude

    def replace_lift_by_drag(self, air_shape_class, engine_weight_class, flight_states):
        # attack of angles is the list and the length is 6
        # 0. warm-up 1. takeoff 2. up 3. cruise 4. down 5. approach
        lift_by_drags = [0] * len(flight_states)
        idx = -1
        for target_flight_state in flight_states:
            idx += 1
            # separate the arguments
            flight_calc_flag = target_flight_state[-1]
            flight_state = target_flight_state[:-1]
            # if attack of angle is zero
            if flight_calc_flag is False:
                continue
            # build the fluid dynamic class
            other_args = air_shape_class.engine_amplitude
            fluid_dynamic_class = FluidDynamic(self.aircraft_name, self.init_mission_class, air_shape_class,
                                               engine_weight_class, flight_state, other_args)

            # calculate lift by drag ratio
            fluid_dynamic_class.calc_lift_by_drag_ratio(self.ld_calc_type)
            lift_by_drags[idx] = fluid_dynamic_class.l_d_cruise

        return lift_by_drags

    # helper function of run meet constraints
    def sizing_of_thrust_mtow_ratio(self, engine_weight_class, air_shape_class, thermal_design_variables, cruise_range):
        # Initialize required variables
        # fix engine weight
        engine_weight = engine_weight_class.total_engine_weight * self.total_mission_class.engine_weight_coef
        # fix electric weight
        electric_weight = engine_weight_class.weight_results[99] * self.total_mission_class.engine_weight_coef
        # fix compressor out blade height of core engine
        co_blade_h = engine_weight_class.co_blade_h * self.total_mission_class.engine_axis_coef
        # fix engine length
        engine_length = engine_weight_class.total_engine_length * self.total_mission_class.engine_length_coef
        # fix front diameter
        front_diameter = engine_weight_class.front_diameter * self.total_mission_class.engine_axis_coef
        # specific fuel consumption
        sfc = engine_weight_class.sfc * self.sfc_coef
        # Velocity of jet engine
        V_jet = engine_weight_class.calc_design_point_class.V_jet

        # aircraft weight ratio
        aircraft_weight_fracs = self.aircraft_weight_ratio
        mass_product = np.prod(aircraft_weight_fracs[:2])
        # by function of set fluid dynamic
        lift_by_drags = self.replace_lift_by_drag(air_shape_class, engine_weight_class, self.flight_states)

        # set the current lift by drag
        # if you want to conduct more detail survey, you have to change this part of code into
        # the formation of dealing with previous aerodynamic results

        if self.ld_form == 'constant':
            cruise_lift_by_drag = self.init_mission_class.Lift_by_Drag
        else:
            cruise_lift_by_drag = lift_by_drags[3]

        # Braguer equations (Cruise state and Loiter state)
        loiter_weight_fracs = np.exp(-self.loiter_time * sfc / cruise_lift_by_drag)
        weight_fracs_cruise = np.exp(-cruise_range * sfc / V_jet / cruise_lift_by_drag / 3.6)
        # Integration loiter
        weight_fracs_cruise = weight_fracs_cruise * loiter_weight_fracs

        aircraft_weight_fracs[3] = weight_fracs_cruise

        # calculate all products of aircraft weight fracs
        weight_fracs_allprod = np.prod(np.array(aircraft_weight_fracs))

        # Converge thrust and max takeoff weight ratio into the target value

        twdiff = 0.0
        twdiffold = 0.0
        current_tw_coef = 1.6  # default
        tw_coef_target = self.baseline_required_thrust_ground / self.baseline_max_takeoff_weight
        # A320 => T/W = 0.289  # (default => 0.3 if mach <1.0 else 0.4)
        # in case of hyper sonic
        if self.init_mission_class.mach > 1.0:
            tw_coef_target = 0.4 * 9.8 * 2

        tw_coef_step = 0.02  # difference of tw coef
        calc_loop = 0  # calculate loop count
        converge_flag = False  # whether calculation converges or not

        # Initialize target thrust
        thrust_target = self.baseline_required_thrust_ground

        # Initialize aircraft weight
        aircraft_weight = self.baseline_aircraft_weight

        # Initialize max takeoff weight
        max_takeoff_weight_new = self.baseline_max_takeoff_weight

        while True:
            # prepare for the list of weight results
            weight_results = [engine_weight, electric_weight, aircraft_weight, max_takeoff_weight_new]

            # calculate amplitude of engine weight
            engine_amplitude = engine_weight / self.baseline_engine_weight

            # determine amplitude of sizing on weight
            sizing_amplitude = self.control_sizing_amplitude(weight_results)
            other_args = sizing_amplitude

            # build aircraft weight class
            aircraft_weight_class = AircraftWeight(self.aircraft_name, self.aircraft_type, engine_weight_class, self.init_mission_class, thermal_design_variables, self.data_base_args, other_args)
            # operate calculation
            aircraft_weight_class.run_airframe()

            # fix aircraft weight
            aircraft_weight = aircraft_weight_class.weight_airframe * self.total_mission_class.air_weight_coef

            if self.control_type == 'engine_add':
                engine_increment = engine_weight - self.baseline_engine_weight
                aircraft_weight += engine_increment * self.engine_alpha

            # calculate max takeoff weight
            w_target = aircraft_weight + self.init_mission_class.payload_weight + engine_weight * \
                       self.init_mission_class.engine_num + electric_weight
            max_takeoff_weight_new = w_target / weight_fracs_allprod

            # calculate difference of indexes
            twdiff = current_tw_coef - tw_coef_target

            # print('twdiff:', twdiff, 'tw_coef:', current_tw_coef, 'tw_coef_target:', tw_coef_target)

            # Error Process
            if np.isnan(twdiff):
                converge_flag = True
                break

            if calc_loop == 100000:
                converge_flag = True
                break

            # judge convergence of thrust max takeoff weight ratio
            if abs(twdiff) < 1.0e-6:
                print('')
                print('=' * 10 + ' Sizing Results ' + '=' * 10)
                print('Cruise SFC:', sfc)
                print('engine weight[kg]:', engine_weight)
                print('aircraft weight[kg]:', aircraft_weight)
                print('max takeoff weight[kg]', max_takeoff_weight_new)
                print('thrust_target[N]:', thrust_target)
                print('Compressor Out Blade Height[m]:', co_blade_h)
                print('Electric Weight[kg]:', electric_weight)
                print('=' * 50)
                break

            # update target thrust and max takeoff weight ratio
            if twdiff * twdiffold < 0.0:
                tw_coef_step *= 0.5

            current_tw_coef += -np.sign(twdiff) * tw_coef_step

            # update target thrust
            thrust_target = current_tw_coef * max_takeoff_weight_new

            twdiffold = twdiff

        return thrust_target, aircraft_weight, engine_weight, max_takeoff_weight_new, weight_fracs_allprod, \
               converge_flag, co_blade_h, electric_weight, cruise_lift_by_drag, mass_product, cruise_range

    # helper function of run meet constraints
    def sizing_of_thrust_mtow_ratio_at_high(self, engine_weight_class, air_shape_class, thermal_design_variables, cruise_range):
        # Initialize required variables
        # fix engine weight
        engine_weight = engine_weight_class.total_engine_weight * self.total_mission_class.engine_weight_coef
        # fix electric weight
        electric_weight = engine_weight_class.weight_results[99] * self.total_mission_class.engine_weight_coef
        # fix compressor out blade height of core engine
        co_blade_h = engine_weight_class.co_blade_h * self.total_mission_class.engine_axis_coef
        # fix engine length
        engine_length = engine_weight_class.total_engine_length * self.total_mission_class.engine_length_coef
        # fix front diameter
        front_diameter = engine_weight_class.front_diameter * self.total_mission_class.engine_axis_coef
        # specific fuel consumption
        sfc = engine_weight_class.sfc_off
        # Velocity of jet engine
        V_jet = engine_weight_class.calc_design_point_class.V_jet

        # aircraft weight ratio
        aircraft_weight_fracs = self.aircraft_weight_ratio
        # by function of set fluid dynamic
        lift_by_drags = self.replace_lift_by_drag(air_shape_class, engine_weight_class, self.flight_states)

        # set the current lift by drag
        # if you want to conduct more detail survey, you have to change this part of code into
        # the formation of dealing with previous aerodynamic results

        if self.ld_form == 'constant':
            cruise_lift_by_drag = self.init_mission_class.Lift_by_Drag
        else:
            cruise_lift_by_drag = lift_by_drags[3]

        # Braguer equations (Cruise state and Loiter state)
        loiter_weight_fracs = np.exp(-self.loiter_time * sfc / cruise_lift_by_drag)
        weight_fracs_cruise = np.exp(-cruise_range * sfc / V_jet / cruise_lift_by_drag / 3.6)
        weight_fracs_cruise = weight_fracs_cruise * loiter_weight_fracs

        aircraft_weight_fracs[3] = weight_fracs_cruise
        mass_product = np.prod(aircraft_weight_fracs[:2])

        # calculate all products of aircraft weight fracs
        weight_fracs_allprod = np.prod(np.array(aircraft_weight_fracs))

        # Converge thrust and max takeoff weight ratio into the target value

        twdiff = 0.0
        twdiffold = 0.0
        current_tw_coef = 1.5  # default
        tw_coef_target = self.baseline_required_thrust_ground / \
                         (self.baseline_max_takeoff_weight - self.init_mission_class.fuel_weight * 0.6)
        # A320 => T/W = 0.289  # (default => 0.3 if mach <1.0 else 0.4)

        tw_coef_step = 0.2  # difference of tw coef
        calc_loop = 0  # calculate loop count
        converge_flag = False  # whether calculation converges or not

        # Initialize target thrust
        thrust_target = self.baseline_required_thrust_ground
        # Initialize aircraft weight
        aircraft_weight = self.baseline_aircraft_weight

        # Initialize max takeoff weight
        max_takeoff_weight_new = self.baseline_max_takeoff_weight

        while True:
            # prepare for the list of weight results
            weight_results = [engine_weight, electric_weight, aircraft_weight, max_takeoff_weight_new]

            # calculate amplitude of engine weight
            engine_amplitude = engine_weight / self.baseline_engine_weight

            # determine amplitude of sizing on weight
            sizing_amplitude = self.control_sizing_amplitude(weight_results)
            other_args = sizing_amplitude

            aircraft_weight_class = AircraftWeight(self.aircraft_name, self.aircraft_type, engine_weight_class,
                                                   self.init_mission_class, thermal_design_variables,
                                                   self.data_base_args,
                                                   other_args)
            # operate calculation
            aircraft_weight_class.run_airframe()

            # fix aircraft weight
            aircraft_weight = aircraft_weight_class.weight_airframe * self.total_mission_class.air_weight_coef

            if self.control_type == 'engine_add':
                engine_increment = engine_weight - self.baseline_engine_weight
                aircraft_weight += engine_increment * self.engine_alpha

            # calculate max takeoff weight
            w_target = aircraft_weight + self.init_mission_class.payload_weight + engine_weight * \
                       self.init_mission_class.engine_num
            max_takeoff_weight_new = w_target / weight_fracs_allprod

            fuel_weight = (1.0 - weight_fracs_allprod) * max_takeoff_weight_new

            # calculate difference of indexes
            twdiff = current_tw_coef - tw_coef_target

            # print('twdiff:', twdiff, 'tw_coef_target:', tw_coef_target)

            # Error Process
            if np.isnan(twdiff):
                converge_flag = True
                break

            if calc_loop == 100000:
                converge_flag = True
                break

            # judge convergence of thrust max takeoff weight ratio
            if abs(twdiff) < 1.0e-5:
                print('')
                print('=' * 10 + ' Sizing Results ' + '=' * 10)
                print('engine weight[kg]:', engine_weight)
                print('aircraft weight[kg]:', aircraft_weight)
                print('max takeoff weight[kg]', max_takeoff_weight_new)
                print('thrust_target[N]:', thrust_target)
                print('Compressor Out Blade Height[m]:', co_blade_h)
                print('Electric Weight[kg]:', electric_weight)
                print('=' * 50)
                break

            # update target thrust and max takeoff weight ratio
            if twdiff * twdiffold < 0.0:
                tw_coef_step *= 0.5

            current_tw_coef += -np.sign(twdiff) * tw_coef_step

            # update target thrust (Assume the cruise state)
            thrust_target = current_tw_coef * (max_takeoff_weight_new - fuel_weight * 0.6)

            twdiffold = twdiff

            # Update operating empty weight
            self.init_mission_class.empty_weight = w_target

            # Update max takeoff weight
            self.init_mission_class.max_takeoff_weight = max_takeoff_weight_new

        return thrust_target, aircraft_weight, engine_weight, max_takeoff_weight_new, weight_fracs_allprod,\
               converge_flag, co_blade_h, electric_weight, cruise_lift_by_drag, mass_product, cruise_range

