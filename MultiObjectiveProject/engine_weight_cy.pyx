import numpy as np
cimport numpy as np
cimport cython
import json
cimport electric_weight_cy
import electric_weight_cy


ctypedef np.float64_t DOUBLE_t


class InitEngineWeightParamsCore(object):

    def __init__(self, str engine_name, str propulsion_type, str engine_data_path):

        self.engine_name = engine_name
        self.propulsion_type = propulsion_type

        # open json file
        f = open(engine_data_path, 'r')
        json_file = json.load(f)
        f.close()
        self.engine_json_file = json_file[self.engine_name]

        self.base_engine_weight = self.engine_json_file['engine_weight']
        self.base_engine_length = self.engine_json_file['engine_length']
        # Number of engine stage
        stage_numbers = self.engine_json_file['stage_number']
        self.fan_stage_number = stage_numbers['Fan']
        self.lpc_stage_number = stage_numbers['LPC']
        self.hpc_stage_number = stage_numbers['HPC']
        self.hpt_stage_number = stage_numbers['HPT']
        self.lpt_stage_number = stage_numbers['LPT']

        # Aspect ratio
        self.aspect_ratio_rotor = 2.2
        self.aspect_ratio_stator = 2.8

        # Tip hub 比
        self.fan_tip_hub_ratio = 0.32
        self.lpc_tip_hub_ratio = 0.5
        self.hpc_tip_hub_ratio = 0.5

        # stage load constant of engine
        stage_coefs = self.engine_json_file['stage_coef']
        self.lpc_stage_coef = stage_coefs['LPC']
        self.hpc_stage_coef = stage_coefs['HPC']
        self.hpt_stage_coef = stage_coefs['HPT']
        self.lpt_stage_coef = stage_coefs['LPT']

        self.u_tip_fan = 350.0
        self.u_tip_lpc = 400.0
        self.u_tip_hpc = 400.0
        self.u_tip_hpt = 480.0
        self.u_tip_lpt = 480.0

        # fan duct length
        self.duct_length = 1.5

        # Design thinking
        # dictionary => (station number: number assigned design thinking)
        if self.propulsion_type in ['turbojet', 'turboshaft', 'TeDP']:

            self.design_thinking = {20: 0, 25: 1, 40: 0, 45: 0}

        elif self.propulsion_type in ['turbofan', 'PartialElectric']:

            self.design_thinking = {10: 0, 20: 0, 25: 1, 40: 0, 45: 0}


class EngineWeight(object):

    def __init__(self, str aircraft_name, str engine_name, str aircraft_type, str propulsion_type, str engine_data_path, calc_design_point_class, calc_off_design_point_class):
        self.aircraft_name = aircraft_name
        self.engine_name = engine_name
        self.aircraft_type = aircraft_type
        self.propulsion_type = propulsion_type

        # class object which has already finished calculation
        self.calc_design_point_class = calc_design_point_class
        self.calc_off_design_point_class = calc_off_design_point_class

        self.fscl = self.calc_design_point_class.fscl
        self.sfc = self.calc_design_point_class.sfc
        self.isp = self.calc_design_point_class.isp

        # Results of design point performances
        self.cref = self.calc_design_point_class.cref
        self.cref_e = self.calc_design_point_class.cref_e
        self.dref = self.calc_design_point_class.dref
        self.qref = self.calc_design_point_class.qref
        self.qref_e = self.calc_design_point_class.qref_e

        # results of off design point performances
        self.coff = self.calc_off_design_point_class.coff
        self.coff_e = self.calc_off_design_point_class.coff_e
        self.doff = self.calc_off_design_point_class.doff
        self.qoff = self.calc_off_design_point_class.qoff
        self.qoff_e = self.calc_off_design_point_class.qoff_e

        # for Distributed fan and Electric Equipment
        design_point_results = [self.cref, self.cref_e, self.dref, self.qref, self.qref_e]
        off_design_point_results = [self.coff, self.coff_e, self.doff, self.qoff, self.qoff_e]

        # Build InitWeightParamsCore class
        self.iewpc = InitEngineWeightParamsCore(self.engine_name, self.propulsion_type, engine_data_path)

        # Build Distributed Fan class
        self.dfw = electric_weight_cy.DistributedFanWeight(design_point_results, off_design_point_results, [self.fscl])

        # Electric density of electric equipment
        electric_component_density = self.calc_design_point_class.electric_component_density
        # Heat Volume of electric part
        self.gene_power = self.calc_off_design_point_class.GenePower
        self.eew = electric_weight_cy.ElectricEquipWeight(design_point_results, off_design_point_results,
                                       [electric_component_density, self.gene_power])

        # Other coefficients
        # Fan
        self.K_fan = 135.0
        self.solidity_fan_ref = 1.25
        self.U_tip_fan_ref = 350.0
        self.aspect_ratio_fan = 2.5

        # Fan Duct
        rou_fan_duct = 2770.0
        thickness_fan_duct = 1.3e-3
        self.duct_wa = rou_fan_duct * thickness_fan_duct

        self.wall_wa = 4.4  # -10.7
        self.splitring_wa = 11.2  # -12.7

        # Compressor
        self.comp_a = 466.0
        self.comp_b = 0.676  # -0.588
        self.comp_c = 0.654e-3

        self.K_comp = 24.2  # if cruise
        # self.K_comp = 15.5 # if lift

        # Combustor
        self.Vref = 18.3  # if cruise
        # self.Vref = 24.4  # if lift

        self.LB_H = 3.2  # if cruise
        # self LB_H = 1.6  # if lift

        self.K_comb = 390.0  # if cruise
        # self.K_comb = 195.0  # if lift

        # Turbine
        # rotor
        self.tur_r_a = 10.45  # aspect ratio coefficient A
        self.tur_r_b = -10.0  # aspect ratio coefficient B

        # stator
        self.tur_s_a = 6.45  # aspect ratio coefficient A
        self.tur_s_b = -5.97  # aspect ratio coefficient B
        self.aT = 0.2  # -1.0 coefficients for calculating the gap distance between each blades

        self.K_tur = 7.9  # if cruise
        # self.K_tur = 4.7  # if lift

        # Control and Accessories
        self.K_acce = 0.0002
        self.a_acce = 13.2

        ############# Engine Results ############
        # Inlet area
        self.inlet_areas = [0] * 100
        # Out area
        self.outlet_areas = [0] * 100

        # Core side component indexes
        self.component_inlets = [20, 25, 30, 40, 45]
        self.component_outlets = [20, 25, 30, 41, 46]

        # All core engine results (less than 100:core weight, more than 100:electric)
        self.weight_results = np.zeros(200)

        # Inlet diameters
        self.inlet_diameter = np.zeros((3, 100))
        # Out diameters
        self.out_diameter = np.zeros((3, 100))

        # Front diameter
        self.front_diameter = None

        # fan max velocity
        FPR = self.dref[22]  # fan pressure ratio at design point
        self.max_fan_tip_velocity = 350 * FPR - 120

        # fan number
        self.Nfan = self.dref[30]

        self.lpt_velocity = 1000

        # Initialize lp or hp shaft revolve rate
        self.lp_shaft_rpm = None
        self.hp_shaft_rpm = None

        # Initialize blade height of outlet compressor
        self.co_blade_h = None

        # Initialize the list of stage numbers
        self.stage_numbers = []

        # Initialize expand coefficients of combustion chamber
        self.expand_cc = 0.2

        # Initialize total engine length
        self.total_engine_length = 0.0

        # Initialize total engine weight
        self.total_engine_weight = 0.0

        # core engine weight
        self.core_engine_weight = 0.0

        # electric weight
        self.electric_weight = 0.0

        # distributed fan weight
        self.distributed_fan_weight = 0.0

        # distributed fan length
        self.distributed_fan_length = 0.0

        # Initialize wide length of distributed fan
        self.distributed_fan_width_length = 0.0

        # distributed fan diameter
        self.distributed_fan_diameter = 0.0

    # main function
    def run_engine(self):

        if self.propulsion_type in ['turbojet', 'turboshaft', 'turbofan']:

            self.run_core_weight()
            # calculate structure and controller weight
            self.calc_structure_and_controller_weight()

        elif self.propulsion_type in ['TeDP', 'PartialElectric']:

            # operate core engine part
            self.run_core_weight()

            # Operate distributed fan calculation
            distributed_fan_length, distributed_fan_width_length, distributed_fan_weight, distributed_fan_duct_weight, fan_in_diameter, fan_out_diameter = self.dfw.run_distributed_fan_weight()


            self.distributed_fan_width_length = distributed_fan_width_length
            self.distributed_fan_length = distributed_fan_length
            self.fan_in_diameter = fan_in_diameter
            self.fan_out_diameter = fan_out_diameter
            self.distributed_fan_diameter = fan_in_diameter[0]

            # insert weight results
            self.weight_results[110] = distributed_fan_weight
            self.weight_results[111] = distributed_fan_duct_weight

            # Operate Electric equipment calculation
            electric_equip_weight = self.eew.run_electric_equip()

            # calculate structure and controller weight
            self.calc_structure_and_controller_weight()

            # Insert weight results
            self.weight_results[99] = electric_equip_weight

        elif self.propulsion_type in ['battery', 'hybridturbojet', 'hybridturbofan']:
            # ToDo: implement battery driving module
            pass

        self.total_engine_weight = np.sum(self.weight_results)
        self.electric_weight = self.weight_results[99]
        self.distributed_fan_weight = self.weight_results[110] + self.weight_results[111]
        self.core_engine_weight = self.total_engine_weight - (self.electric_weight + self.distributed_fan_weight)

        # if self.lpt_velocity >= self.iewpc.u_tip_lpt:
        # self.total_engine_weight = np.nan

        print('')
        print('=' * 10 + 'FINAL ENGINE RESULTS' + '=' * 10)
        print('distributed fan diameter:', self.distributed_fan_diameter)
        print('Total Engine Weight:', self.total_engine_weight)
        print('Core Engine Weight:', self.core_engine_weight, 'Electric Weight:', self.electric_weight, 'Distributed Fan Weight:', self.distributed_fan_weight)
        print('')

    # Operate core engine function
    def run_core_weight(self):
        # Before starting calculating weight module, we have to finish calculating thermal analysis

        # Calculate Cross Area
        self.calc_area()

        # Calculate cross Diameter
        self.calc_diameter()

        # Calculate stage number
        self.calc_stage_number()

        # calculate weight of core
        self.calc_core_weight()

    # calculate cross area of each components
    def calc_area(self):

        for idx in self.component_inlets:
            self.inlet_areas[idx] = self.qref[2, idx] * self.fscl

        for idx in self.component_outlets:
            self.outlet_areas[idx] = self.qref[11, idx] * self.fscl

    # calculate diameters of each components
    def calc_diameter(self):

        # Determine the no constraints of diameters
        def comp_diameter_no_constraints(component_index, tip_hub_ratio, areas):
            """

            :param component_index: int (ex. 'HPC' = 25)
            :param tip_hub_ratio: float
            :param d_think: int (0 - 2)
            :param areas: list (all component's areas)
            :return: diams( list of results of diameters)
            """

            d_tip = 2.0 * np.sqrt(areas[component_index] / (np.pi * (1.0 - tip_hub_ratio ** 2)))
            d_hub = d_tip * tip_hub_ratio
            d_mid = 0.5 * (d_tip + d_hub)

            diams = [d_tip, d_mid, d_hub]

            return diams

        # Determine the constraints of diameters
        def comp_diameter_constraints(component_index, diams, d_think, areas):

            """

            :param component_index: int
            :param diams: list (previous results of diameter)
            :param d_think: int
            :param areas: list (all results of each component's area)
            :return: next_diams
            """

            # Set the three types of diameter (Tip, mid, hub)
            d_tip, d_mid, d_hub = diams

            # Inside diameter constant
            if d_think == 0:
                d_hub = d_hub
                d_tip = np.sqrt(d_hub ** 2 + 4.0 / np.pi * areas[component_index])
                d_mid = 0.5 * (d_tip + d_hub)

            # Outside diameter constant
            elif d_think == 1:
                d_tip = d_tip
                d_hub = np.sqrt(d_tip ** 2 - 4.0 / np.pi * areas[component_index])
                d_mid = 0.5 * (d_tip + d_hub)

            elif d_think == 2:
                d_mid = d_mid
                # Cross section diameter
                area_diam = np.sqrt(4.0 / np.pi * areas[component_index])
                d_tip = d_mid + 0.5 * area_diam
                d_hub = d_mid - 0.5 * area_diam

            else:

                return []

            next_diams = [d_tip, d_mid, d_hub]

            return next_diams


        # fan
        # Inlet
        A10 = self.qref[2, 10] * self.fscl
        d_tip = 2.0 * np.sqrt(A10 / (1.0 - self.iewpc.fan_tip_hub_ratio ** 2) / np.pi)
        d_hub = d_tip * self.iewpc.fan_tip_hub_ratio
        d_mid = 0.5 * (d_tip + d_hub)
        front_diam = d_tip
        self.lp_shaft_rpm = 30 * self.max_fan_tip_velocity / (np.pi * 0.5 * front_diam)
        # print('lpt shaft param:', self.lp_shaft_rpm)

        fan_in_diams = [d_tip, d_mid, d_hub]

        # Results (Inlet)
        for idx, diam in enumerate(fan_in_diams):
            self.inlet_diameter[idx, 10] = diam

        # Outlet
        A19 = self.qref[2, 19] * self.fscl
        d_tip = d_tip
        d_hub = np.sqrt(d_tip ** 2 - 4.0 * (A19) / np.pi)
        d_mid = 0.5 * (d_tip + d_hub)

        fan_out_diams = [d_tip, d_mid, d_hub]

        for idx, diam in enumerate(fan_out_diams):
            self.out_diameter[idx, 10] = diam

        ########################### Compressor Part #########################

        # lpc
        # Inlet Calculation
        if self.propulsion_type in ['turbojet', 'turboshaft', 'TeDP']:
            A20 = self.qref[2, 20] * self.fscl
            d_tip = 2.0 * np.sqrt(A20 / (1.0 - self.iewpc.lpc_tip_hub_ratio ** 2) / np.pi)
            self.lp_shaft_rpm = 30 * self.iewpc.u_tip_lpc / (0.5 * d_tip * np.pi)
        elif self.propulsion_type in ['turbofan', 'PartialElectric']:
            d_tip = self.iewpc.u_tip_lpc * 60 / (np.pi * self.lp_shaft_rpm)

        # print('lp shaft rpm:', self.lp_shaft_rpm, d_tip)

        d_hub = d_tip * self.iewpc.lpc_tip_hub_ratio
        d_mid = 0.5 * (d_tip + d_hub)
        lpc_in_diams = [d_tip, d_mid, d_hub]
        # print('lpc in diam:', lpc_in_diams)

        # Results (Inlet)
        for idx, diam in enumerate(lpc_in_diams):
            self.inlet_diameter[idx, 20] = diam

        # Outlet calculation
        d_think = self.iewpc.design_thinking[20]

        lpc_out_diams = comp_diameter_constraints(20, lpc_in_diams, d_think, self.outlet_areas)
        # print('lpc out diam:', lpc_out_diams)

        # Results (Out)
        for idx, diam in enumerate(lpc_out_diams):
            self.out_diameter[idx, 20] = diam

        # hpc
        # Inlet Calculation
        hpc_in_diams = comp_diameter_no_constraints(25, self.iewpc.hpc_tip_hub_ratio, self.inlet_areas)

        # HP shaft revolve rate (rpm)
        self.hp_shaft_rpm = 30 * self.iewpc.u_tip_hpc / (np.pi * 0.5 * hpc_in_diams[0])
        # print('hp shaft rpm:', self.hp_shaft_rpm)
        # print('hpc in diam:', hpc_in_diams)

        # Results (Inlet)
        for idx, diam in enumerate(hpc_in_diams):
            self.inlet_diameter[idx, 25] = diam

        # Outlet Calculation
        d_think = self.iewpc.design_thinking[25]

        hpc_out_diams = comp_diameter_constraints(25, hpc_in_diams, d_think, self.outlet_areas)
        # print('hpc out diam:', hpc_out_diams)

        # Results (Out)

        for idx, diam in enumerate(hpc_out_diams):
            self.out_diameter[idx, 25] = diam

        # Blade Height of Out Compressor
        self.co_blade_h = (self.out_diameter[0, 25] - self.out_diameter[2, 25]) * 0.5
        # print('co blade height:', self.co_blade_h)

        # CC
        dcc_tip = (1.0 + self.expand_cc * 0.5) * hpc_out_diams[0]
        dcc_hub = (1.0 - self.expand_cc * 0.5) * hpc_out_diams[2]
        dcc_mid = 0.5 * (dcc_tip + dcc_hub)

        cc_in_diams = [dcc_tip, dcc_mid, dcc_hub]

        # Results (Inlet)
        for idx, diam in enumerate(cc_in_diams):
            self.inlet_diameter[idx, 30] = diam

        ########################### Turbine Part #############################

        # This part is that former calculation is out, while latter calculation is inlet

        def turb_diameter_no_constraints(component_index, tip_vel, shaft_rpm, areas):
            """

            :param component_index: int
            :param tip_vel: float (Component max revolving speed (m/s))
            :param shaft_rpm: float (rpm)
            :param areas: list
            :return: list (results of diameters)
            """

            d_tip = 30.0 * tip_vel / (np.pi * shaft_rpm * 0.5)
            d_hub = np.sqrt(d_tip ** 2 - 4.0 * areas[component_index] / np.pi)
            d_mid = 0.5 * (d_tip + d_hub)


            diams = [d_tip, d_mid, d_hub]

            return diams

        def turb_diameter_constraints(component_index, diams, d_think, areas):
            """

            :param component_index: int
            :param diams: list (previous results of each component's diameter)
            :param d_think: int
            :param areas: list
            :return: list (results of diameters)
            """
            d_tip, d_mid, d_hub = diams

            # Inside diameter constant
            if d_think == 0:
                d_hub = d_hub
                d_tip = np.sqrt(d_hub ** 2 + 4.0 / np.pi * areas[component_index])
                d_mid = 0.5 * (d_tip + d_hub)

            # Outside diameter constant
            elif d_think == 1:
                d_tip = d_tip
                d_hub = np.sqrt(d_tip ** 2 - 4.0 / np.pi * areas[component_index])
                d_mid = 0.5 * (d_tip + d_hub)

            elif d_think == 2:
                d_mid = d_mid
                # Cross section diameter
                area_diam = np.sqrt(4.0 / np.pi * areas[component_index])
                d_tip = d_mid + 0.5 * area_diam
                d_hub = d_mid - 0.5 * area_diam

            else:

                return []

            next_diams = [d_tip, d_mid, d_hub]

            return next_diams

        # hpt
        # Out
        hpt_out_diams = turb_diameter_no_constraints(41, self.iewpc.u_tip_hpt, self.hp_shaft_rpm,
                                                     self.outlet_areas)

        # Results (Out)
        for idx, diam in enumerate(hpt_out_diams):
            self.out_diameter[idx, 41] = diam

        # Inlet
        d_think = self.iewpc.design_thinking[40]

        hpt_in_diams = turb_diameter_constraints(40, hpt_out_diams, d_think, self.inlet_areas)

        # print('hpt in diam:', hpt_in_diams)
        # print('hpt out diam:', hpt_out_diams)

        # Results (Inlet)
        for idx, diam in enumerate(hpt_in_diams):
            self.inlet_diameter[idx, 40] = diam

        # lpt
        # Out
        d_hub = hpt_out_diams[2]
        d_tip = np.sqrt(d_hub ** 2 + 4.0 / np.pi * self.inlet_areas[45])
        d_mid = 0.5 * (d_tip + d_hub)
        lpt_out_diams = [d_tip, d_mid, d_hub]
        # calculate lpt velocity
        self.lpt_velocity = np.pi * self.lp_shaft_rpm * d_tip / 60

        # Results (Out)
        for idx, diam in enumerate(lpt_out_diams):
            self.out_diameter[idx, 46] = diam

        # Inlet
        d_think = self.iewpc.design_thinking[45]
        lpt_in_diams = turb_diameter_constraints(40, lpt_out_diams, d_think, self.inlet_areas)

        # print('lpt in diam:', lpt_in_diams)
        # print('lpt out diam:', lpt_out_diams)

        # Results (Inlet)
        for idx, diam in enumerate(lpt_in_diams):
            self.inlet_diameter[idx, 45] = diam

        # print('')
        # print('lpt velocity:', self.lpt_velocity)
        # print('')

    # calculate stage number
    def calc_stage_number(self):

        def each_stage_number(component_indexes, shaft_rpm, stage_coef, diameters, thermal_results):
            """

            :param component_indexes: tuple (inlet index, out index)
            :param shaft_rpm: float
            :param stage_coef: float (coefficient of stage load)
            :param diameters: list [inlet diameter, out diameter]
            :param thermal_results: list [cref, dref, qref]
            :return: float (stage number)
            """
            # Set component index
            cur_cidx, next_cidx = component_indexes

            # Set physical condition
            cref, dref, qref = thermal_results

            # Set diameters
            in_diam, out_diam = diameters

            T_in = qref[4, cur_cidx]
            T_out = qref[4, next_cidx]
            cp = cref[1, cur_cidx]

            # Calculate Mean velocity
            Um1 = in_diam[1, cur_cidx] * np.pi * shaft_rpm / 60.0
            if cur_cidx > 30:
                Umn = out_diam[1, next_cidx] * np.pi * shaft_rpm / 60.0
            else:
                Umn = out_diam[1, cur_cidx] * np.pi * shaft_rpm / 60.0

            Um = 0.5 * (Um1 + Umn)

            target = (cp * abs(T_in - T_out)) / (stage_coef * Um ** 2)

            # print('current index:', cur_cidx)
            # print('Tin:', T_in, 'Tout:', T_out)
            # print('Um:', Um, 'Um1:', Um1, 'Umn:', Umn, 'stage_number:', target)
            return target

        # local global variables
        thermal_results = [self.cref, self.dref, self.qref]
        diameters = [self.inlet_diameter, self.out_diameter]

        # lpc
        lpc_stage_number = each_stage_number((20, 25), self.lp_shaft_rpm, self.iewpc.lpc_stage_coef, diameters,
                                             thermal_results)
        self.iewpc.lpc_stage_number = lpc_stage_number

        # hpc
        hpc_stage_number = each_stage_number((25, 30), self.hp_shaft_rpm, self.iewpc.hpc_stage_coef, diameters,
                                             thermal_results)
        self.iewpc.hpc_stage_number = hpc_stage_number

        # hpt
        hpt_stage_number = each_stage_number((40, 41), self.hp_shaft_rpm, self.iewpc.hpt_stage_coef, diameters,
                                             thermal_results)
        self.iewpc.hpt_stage_number = hpt_stage_number

        # lpt
        lpt_stage_number = each_stage_number((45, 46), self.lp_shaft_rpm, self.iewpc.lpt_stage_coef, diameters,
                                             thermal_results)
        self.iewpc.lpt_stage_number = lpt_stage_number

        print('')
        print('=' * 10 + 'engine stage number' + '=' * 10)
        print('lpc stage number:', lpc_stage_number)
        print('hpc stage_number:', hpc_stage_number)
        print('hpt stage number:', hpt_stage_number)
        print('lpt stage number:', lpt_stage_number)
        print('')

        self.stage_numbers = [lpc_stage_number, hpc_stage_number, hpt_stage_number, lpt_stage_number]

    # Calculate Core side of total weight
    def calc_core_weight(self):

        ################################ Fan (Core) Part ##########################
        # calculate length
        cxr = (self.inlet_diameter[0, 10] - self.inlet_diameter[2, 10]) / (2.0 * self.iewpc.aspect_ratio_rotor)
        cxs = (self.out_diameter[0, 10] - self.out_diameter[2, 10]) / (2.0 * self.iewpc.aspect_ratio_stator)

        core_fan_length = cxr * (1.0 + 2) + cxs

        # Real tip velocity with Fan Pressure ratio
        U_tip = self.lp_shaft_rpm * np.pi * self.inlet_diameter[0, 10] / 60.0

        # weight fan
        weight_fan = self.K_fan * (self.inlet_diameter[0, 10] ** 2.7 / (self.aspect_ratio_fan) ** 0.5) * \
                     (self.solidity_fan_ref / self.solidity_fan_ref) ** 0.3 * (U_tip / self.U_tip_fan_ref) ** 0.3

        # weight fan duct
        weight_fan_duct = 8.0 * np.pi * (self.inlet_diameter[0, 10] + self.out_diameter[0, 10]) * self.iewpc.duct_length

        self.weight_results[10] = weight_fan
        self.weight_results[11] = weight_fan_duct

        # set the core front diameter
        if self.propulsion_type in ['turbojet', 'turboshaft', 'TeDP']:
            self.front_diameter = self.inlet_diameter[0, 20]

        elif self.propulsion_type in ['turbofan', 'PartialElectric']:
            self.front_diameter = self.inlet_diameter[0, 10]

        # Overall fan length
        fan_length = core_fan_length + self.iewpc.duct_length

        ########################### Compressor Part ##############################

        # help to calculate compressor's length and weight
        def compressor_indexes(component_index, stage_number, diameters, tip_hub_ratio, an_coefs):
            """

            :param component_index: int
            :param stage_number: int
            :param diameters: [inlet_diameters, out_diameters]
            :param tip_hub_ratio: float
            :param an_coefs: [K_comp, u_tip_lpc, u_tip_lpc_ref]
            :return: length, weight
            """

            # set diameters
            in_diams, out_diams = diameters

            # set coefficients
            K_comp, u_tip_lpc, u_tip_lpc_ref = an_coefs

            mean_diam = 0.5 * (in_diams[0, component_index] + out_diams[1, component_index])

            length_coef = 1.0 + (0.2 + (0.234 - 0.218 * tip_hub_ratio) * stage_number) / (0.2 + 0.081 * stage_number)

            weight = K_comp * (mean_diam ** 2.2) * (stage_number ** 1.2) * length_coef * (
                    u_tip_lpc / u_tip_lpc_ref) ** 0.5

            length = mean_diam * (0.2 + (0.234 - 0.218 * tip_hub_ratio) * stage_number)

            return length, weight

        # local global variables
        diameters = [self.inlet_diameter, self.out_diameter]

        # lpc
        lpc_stage_number = self.iewpc.lpc_stage_number
        lpc_an_coefs = [self.K_comp, self.iewpc.u_tip_lpc, self.U_tip_fan_ref]

        # calculate lpc component length and weight
        lpc_length, lpc_weight = compressor_indexes(20, lpc_stage_number, diameters, self.iewpc.lpc_tip_hub_ratio,
                                                    lpc_an_coefs)

        self.weight_results[20] = lpc_weight

        # hpc
        hpc_stage_number = self.iewpc.hpc_stage_number
        hpc_an_coefs = [self.K_comp, self.iewpc.u_tip_hpc, self.U_tip_fan_ref]

        # calculate hpc component length and weight
        hpc_length, hpc_weight = compressor_indexes(25, hpc_stage_number, diameters, self.iewpc.hpc_tip_hub_ratio,
                                                    hpc_an_coefs)

        self.weight_results[25] = hpc_weight

        ########################### Turbine Part ############################

        # help to calculate turbine weight and length
        def turbine_indexes(component_indexes, stage_number, diameters, tip_velocity, an_coefs):

            # set component index
            in_cidx, out_cidx = component_indexes
            # set diameters
            in_diams, out_diams = diameters

            # set coefficients
            K_tur, tur_sta_coef, tur_rot_coef, aT = an_coefs

            # Tip, mid, hub diameters
            all_part_diam = 0.5 * (in_diams[:, in_cidx] + out_diams[:, out_cidx])

            tip_diam, mean_diam, hub_diam = all_part_diam.tolist()

            # calculate weight
            weight = K_tur * (mean_diam ** 2.5) * stage_number * (tip_velocity ** 0.6)

            # Length Part
            AR_stator = tur_sta_coef[0] + tur_sta_coef[1] * (hub_diam / tip_diam)
            AR_rotor = tur_rot_coef[0] + tur_rot_coef[1] * (hub_diam / tip_diam)

            # calculate requiring coefficients
            cxr = (tip_diam - hub_diam) / (2.0 * AR_rotor)
            cxs = (tip_diam - hub_diam) / (2.0 * AR_stator)
            st = aT * cxr

            # calculate length
            length = stage_number * (cxr + cxs) + (2.0 * stage_number - 1) * st

            return length, weight

        # hpt
        hpt_stage_number = self.iewpc.hpt_stage_number
        hpt_an_coefs = [self.K_tur, [self.tur_s_a, self.tur_s_b], [self.tur_r_a, self.tur_r_b], self.aT]
        hpt_length, hpt_weight = turbine_indexes((40, 41), hpt_stage_number, diameters, self.iewpc.u_tip_hpt,
                                                 hpt_an_coefs)

        self.weight_results[40] = hpt_weight

        ########## Combustion Chamber Part ###########
        h30 = (self.out_diameter[0, 25] - self.out_diameter[2, 25]) * 0.5

        dcc_mid_in = self.out_diameter[1, 25]
        dcc_mid_out = self.out_diameter[1, 40]
        mean_diam = 0.5 * (dcc_mid_in + dcc_mid_out)
        # set thermal variables
        rg30 = self.cref[1, 30] / (self.cref[2, 25] / (self.cref[2, 25] - 1))
        # Airflow rate
        W30 = self.qref[1, 30]
        # Total Temperature and Pressure
        TT30 = self.qref[4, 30]
        PT30 = self.qref[5, 30]

        LB = rg30 / (np.pi * self.Vref) * (self.LB_H * W30 * TT30 / PT30 / mean_diam)
        HCC = LB / self.LB_H

        # calculate diameters of cc at each place
        # Inlet
        dcc_tip_in = dcc_mid_in + HCC
        dcc_hub_in = dcc_mid_in - HCC

        cc_in_diams = [dcc_tip_in, dcc_mid_in, dcc_hub_in]

        # Out
        dcc_tip_out = dcc_mid_out + HCC
        dcc_hub_out = dcc_mid_out - HCC

        cc_out_diams = [dcc_tip_out, dcc_mid_out, dcc_hub_out]

        # Results
        for idx, diam in enumerate(cc_in_diams):
            self.inlet_diameter[idx, 30] = diam

        for idx, diam in enumerate(cc_out_diams):
            self.out_diameter[idx, 30] = diam

        # calculate combustion chamber weight
        cc_weight = self.K_comb * mean_diam ** 2.0
        cc_length = LB

        self.weight_results[30] = cc_weight

        # lpt
        lpt_stage_number = self.iewpc.lpt_stage_number
        lpt_an_coefs = [self.K_tur, [self.tur_s_a, self.tur_s_b], [self.tur_r_a, self.tur_r_b], self.aT]
        lpt_length, lpt_weight = turbine_indexes((45, 46), lpt_stage_number, diameters, self.iewpc.u_tip_lpt,
                                                 lpt_an_coefs)

        self.weight_results[45] = lpt_weight

        # calculate total engine length
        self.total_engine_length = fan_length + lpc_length + hpc_length + hpt_length + cc_length + lpt_length

    def calc_structure_and_controller_weight(self):

        # subtract electric weight from total weight results
        other_weight_sum = np.sum(self.weight_results) - self.weight_results[99]

        # controller weight
        controller_weight = other_weight_sum * 10.0 / 75.0

        # structure weight
        structure_weight = other_weight_sum * 15.0 / 75.0

        self.weight_results[91] = controller_weight
        self.weight_results[92] = structure_weight




# calculate engine weight function
cpdef calc_engine_weight(list args):
    cdef:
        str aircraft_name
        str engine_name
        str aircraft_type
        str propulsion_type
        str engine_data_path

    aircraft_name, engine_name, aircraft_type, propulsion_type, engine_data_path, calc_design_point_class, calc_off_design_point_class = args
    engine_weight_class = EngineWeight(aircraft_name, engine_name, aircraft_type, propulsion_type, engine_data_path, calc_design_point_class, calc_off_design_point_class)
    engine_weight_class.run_engine()

    return engine_weight_class








