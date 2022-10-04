import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float64_t DOUBLE_t


# Calculate Distributed Fan weight
cdef class DistributedFanWeight(object):

    def __init__(self, list design_point_results, list off_design_point_results, list other_args_list):
        """

        :param design_point_results: list ([cref, cref_e, dref, qref, qref_e][
        :param off_design_point_results: list ([coff, coff_e, doff, qoff, qoff_e])
        :param other_args: list ([fscl])
        """
        # Design point and Design off point results
        self.cref, self.cref_e, self.dref, self.qref, self.qref_e = design_point_results
        self.coff, self.coff_e, self.doff, self.qoff, self.qoff_e = off_design_point_results

        # set other arguments
        if len(other_args_list) == 1:
            other_args = other_args_list[0]
        self.fscl = other_args

        # Engine components variables
        self.Kf = 135.0  # weight coefficient of Fan
        self.AR = 2.5  # Aspect ratio
        self.sigma_t = 1.25  # solidity
        self.sigma_t_ref = 1.25  # solidity reference
        self.u_tip_ref = 350.0
        self.density = 8.0  # density of aluminium (kg/m**3)

        self.duct_length = 1.5
        # fan number
        self.Nfan = self.dref[30]

        # fan tip hub ratio
        self.fan_tip_hub_ratio = 0.32

        # aspect ratio
        self.aspect_ratio_rotor = 2.2
        self.aspect_ratio_stator = 2.8

        # Initialize weight variables
        self.distributed_fan_weight = 0.0
        self.distributed_fan_duct_weight = 0.0
        self.distributed_fan_length = 0.0
        self.distributed_fan_width_length = 0.0

        # distributed fan results
        self.fan_in_diameter = [0] * 3
        self.fan_out_diameter = [0] * 3

    def run_distributed_fan_weight(self):

        print('')
        print('=' * 10 + ' Distributed fan configuration ' + '=' * 10)

        weight_coefs = [self.Kf, self.AR, self.sigma_t, self.sigma_t_ref, self.u_tip_ref, self.duct_length]
        args = [self.dref, self.qref_e, self.fscl, self.Nfan, self.fan_tip_hub_ratio, self.aspect_ratio_rotor, self.aspect_ratio_stator, weight_coefs, self.fan_in_diameter, self.fan_out_diameter]

        self.distributed_fan_length, self.distributed_fan_width_length, self.distributed_fan_weight, self.distributed_fan_duct_weight, self.fan_in_diameter, self.fan_out_diameter = calc_distributed_fan(args)

        """
        # Front area of distributed fan
        AE10 = self.qref_e[2, 10] * self.fscl

        # Inlet
        d_tip = 2.0 * np.sqrt((AE10 / self.Nfan) / (1.0 - self.fan_tip_hub_ratio ** 2) / np.pi)
        d_hub = d_tip * self.fan_tip_hub_ratio
        d_mid = 0.5 * (d_tip + d_hub)

        dist_fan_in_diams = [d_tip, d_mid, d_hub]

        # results (Inlet)
        for idx, diam in enumerate(dist_fan_in_diams):
            self.fan_in_diameter[idx] = diam

        # Out (Design thinking : Outside diameter constant)

        # out fan area
        AE19 = self.qref_e[2, 19]
        d_tip = d_tip
        d_hub = np.sqrt(d_tip ** 2 - 4.0 * (AE19 / self.Nfan) / np.pi)
        d_mid = 0.5 * (d_tip + d_hub)

        dist_fan_out_diams = [d_tip, d_mid, d_hub]

        # results (Out)
        for idx, diam in enumerate(dist_fan_out_diams):
            self.fan_out_diameter[idx] = diam

        # calculate length
        cxr = (self.fan_in_diameter[0] - self.fan_in_diameter[2]) / (2.0 * self.aspect_ratio_rotor)
        cxs = (self.fan_out_diameter[0] - self.fan_out_diameter[2]) / (2.0 * self.aspect_ratio_stator)

        self.distributed_fan_length = cxr * (1.0 + 2) + cxs

        FPRe = self.dref[32]  # Fan Pressure Ratio of distributed fan
        U_tip = 350.0 * FPRe - 120.0  # Tip velocity of distributed fan

        # calculate weight
        self.distributed_fan_weight = self.Kf * (self.fan_in_diameter[0]) ** 2.7 / (self.AR) ** 0.5 * \
                               (self.sigma_t / self.sigma_t_ref) ** 0.3 * (U_tip / self.u_tip_ref) ** 0.3 * self.Nfan

        # duct
        self.distributed_fan_duct_weight = 8.0 * np.pi * (self.fan_in_diameter[0] + self.fan_out_diameter[0]) * self.duct_length * self.Nfan

        self.distributed_fan_length = self.distributed_fan_length + self.duct_length

        # width length of distributed fan
        self.distributed_fan_width_length = self.fan_in_diameter[0] * self.Nfan
        """

        return self.distributed_fan_length, self.distributed_fan_width_length, self.distributed_fan_weight, self.distributed_fan_duct_weight, self.fan_in_diameter, self.fan_out_diameter



# Calculate Electric Equipment weight (ex. Generator, Batteries)
cdef class ElectricEquipWeight(object):

    def __init__(self, list design_point_results, list off_design_point_results, list other_args_list, battery_types = None):
        """

        :param design_point_results: list ([cref, cref_e, dref, qref, qref_e][
        :param off_design_point_results: list ([coff, coff_e, doff, qoff, qoff_e])
        :param other_args: list ([electric_component_density, gene_power])
        :param battery_types: dictionary {'polar_material': , 'polar_material_mol_mass':, , 'polar_material_density':,
        'electron_ratio': , 'volta':,}
        """

        # Design point and Design off point results
        self.cref, self.cref_e, self.dref, self.qref, self.qref_e = design_point_results
        self.coff, self.coff_e, self.doff, self.qoff, self.qoff_e = off_design_point_results

        # set other arguments
        if len(other_args_list) == 1:
            other_args = other_args_list[0]
        self.electric_component_density, self.gene_power = other_args_list

        self.battery_types = battery_types

        # Battery coefficients
        self.k_cw = 26.8  # (Ah/mol)

        # E;lectric Equipment weight
        self.electric_equip_weight = 0.0
        self.generator_weight = 0.0
        self.battery_weight = 0.0

    def run_electric_equip(self):

        if self.battery_types is not None:

            self.calc_battery()

        self.calc_generator()

        self.electric_equip_weight = self.battery_weight + self.generator_weight

        return self.electric_equip_weight

    # Calculate weight of Generator
    def calc_generator(self):

        self.generator_weight = self.gene_power * self.electric_component_density  # [kg]

    # Calculate weight of Battery
    def calc_battery(self):
        # 1J = 1C * 1V
        # 1C = 1/ 3600 (Ah)
        # Ne : 電子のモル数 M: 活物質のモル質量（リチウムイオン電池ならば、LiCo2)
        # Cw (Ah/g) = 26.8 * Ne / M

        # ratio of electron (mol)
        Ne = self.battery_types['electron_ratio']
        # mol mass
        M = self.battery_types['polar_material_mol_mass']
        # density
        density = self.battery_types['polar_material_density']
        # volta
        volta = self.battery_types['volta']  # 300V ~ 400V

        # per weight(g)
        cw = self.k_cw * Ne / M  # (Ah/g)
        # per volume(m**3)
        cv = cw * density  # (Ah/m**3)

        # max entarpy
        L_ele = self.gene_power * 1.0e+3

        # Ah
        ah = L_ele / volta * 3600

        self.battery_weight = ah / cw



cdef calc_distributed_fan(list args):
    cdef:
        np.ndarray[DOUBLE_t, ndim=1] dref
        np.ndarray[DOUBLE_t, ndim=2] qref_e
        double fscl
        double Nfan
        double fan_tip_hub_ratio
        double aspect_ratio_rotor
        double aspect_ratio_stator
        list weight_coefs
        list fan_in_diameter
        list fan_out_diameter
        double Kf
        double AR
        double sigma_t
        double sigma_t_ref
        double u_tip_ref
        double duct_length
        double AE10
        double d_tip
        double d_hub
        double d_mid
        list dist_fan_in_diams
        int idx
        double diam
        double AE19
        list dist_fan_out_diams
        double cxr
        double cxs
        double distributed_fan_length
        double FPRe
        double U_tip
        double distributed_fan_weight
        double distributed_fan_duct_weight
        double distributed_fan_width_length
        list results

    dref, qref_e, fscl, Nfan, fan_tip_hub_ratio, aspect_ratio_rotor, aspect_ratio_stator, weight_coefs, fan_in_diameter, fan_out_diameter = args

    Kf, AR, sigma_t, sigma_t_ref, u_tip_ref, duct_length = weight_coefs

    # Front area of distributed fan
    AE10 = qref_e[2, 10] * fscl

    # Inlet
    front_in_diam = 2.0 * np.sqrt(AE10 / (1.0 - fan_tip_hub_ratio ** 2) / np.pi)
    d_tip = 2.0 * np.sqrt((AE10 / Nfan) / (1.0 - fan_tip_hub_ratio ** 2) / np.pi)
    d_hub = d_tip * fan_tip_hub_ratio
    d_mid = 0.5 * (d_tip + d_hub)

    dist_fan_in_diams = [d_tip, d_mid, d_hub]

    # results (Inlet)
    for idx, diam in enumerate(dist_fan_in_diams):
        fan_in_diameter[idx] = diam

    # Out (Design thinking : Outside diameter constant)

    # out fan area
    AE19 = qref_e[2, 19]
    d_tip = d_tip
    d_hub = np.sqrt(d_tip ** 2 - 4.0 * (AE19 / Nfan) / np.pi)
    d_mid = 0.5 * (d_tip + d_hub)

    dist_fan_out_diams = [d_tip, d_mid, d_hub]

    # results (Out)
    for idx, diam in enumerate(dist_fan_out_diams):
        fan_out_diameter[idx] = diam

    # calculate length
    cxr = (fan_in_diameter[0] - fan_in_diameter[2]) / (2.0 * aspect_ratio_rotor)
    cxs = (fan_out_diameter[0] - fan_out_diameter[2]) / (2.0 * aspect_ratio_stator)

    distributed_fan_length = cxr * (1.0 + 2) + cxs

    FPRe = dref[32]  # Fan Pressure Ratio of distributed fan
    U_tip = 350.0 * FPRe - 120.0  # Tip velocity of distributed fan

    # calculate weight
    distributed_fan_weight = Kf * (fan_in_diameter[0]) ** 2.7 / (AR) ** 0.5 * \
                               (sigma_t / sigma_t_ref) ** 0.3 * (U_tip / u_tip_ref) ** 0.3 * Nfan

    # duct
    distributed_fan_duct_weight = 8.0 * np.pi * (fan_in_diameter[0] + fan_out_diameter[0]) * duct_length * Nfan * fan_in_diameter[0] / front_in_diam

    distributed_fan_length = distributed_fan_length + duct_length

    # width length of distributed fan
    distributed_fan_width_length = fan_in_diameter[0] * Nfan

    results = [distributed_fan_length, distributed_fan_width_length, distributed_fan_weight, distributed_fan_duct_weight, fan_in_diameter, fan_out_diameter]

    return results



