import numpy as np
from init_airparams import InitAirParams
from bwb_weight import BlendedWingBodyWeight

# Aircraft Component class
# Main purpose is calculating component's weight

# Main Wing
class MainWing(InitAirParams):
    """
    aaa
    """

    def __init__(self, aircraft_name, aircraft_type, init_mission_class, engine_amplitude):
        self.module_index = 1
        self.aircraft_type = aircraft_type
        self.aircraft_name = aircraft_name

        super().__init__(self.aircraft_name, init_mission_class, engine_amplitude)

        # module name
        self.name = self.module_dict[self.module_index]

    def set_bwb_config(self, blended_wing_body_class):
        """

        :param blended_wing_body_class: Already built and calculated
        :return:
        """
        # Exposed wing area
        self.Sw = blended_wing_body_class.bwb_Sw

    def __call__(self, weight_results, other_args):
        """

        :param weight_results: ndarray
        :param other_args: [max_takeoff_weight]
        :return:
        """
        # Max takeoff weight
        max_takeoff_weight = other_args[0]
        # from kg to lb
        max_takeoff_weight *= self.kg_to_lb
        # (1) wing weight
        cosa = np.cos(self.theta * np.pi / 180.0)

        # calculate wing weight
        W_wing = 5.1e-3 * ((max_takeoff_weight * self.Nz) ** 0.557) * (self.Sw ** 0.649) * (self.AR ** 0.5) * (
                self.tc ** (-0.4) * ((1.0 + self.Lambda) ** 0.1) / cosa * (self.Scsw ** 0.1))

        weight_results[self.module_index] = W_wing

        return weight_results


# Horizontal wing
class HorizontalWing(InitAirParams):

    def __init__(self, aircraft_name, aircraft_type, init_mission_class, engine_amplitude):
        self.aircraft_name = aircraft_name
        self.aircraft_type = aircraft_type

        super().__init__(self.aircraft_name, init_mission_class, engine_amplitude)

        self.module_index = 2

        # module name
        self.name = self.module_dict[self.module_index]

    def __call__(self, weight_results, other_args):
        """

        :param weight_results: ndarray
        :param other_args: [max_takeoff_weight]
        :return:
        """
        # Max takeoff weight
        max_takeoff_weight = other_args[0]
        # from kg to lb
        max_takeoff_weight *= self.kg_to_lb

        # retreating angle
        cosT = np.cos(self.theta_h * np.pi / 180.0)

        # calculate horizontal wing
        W_ht = 3.79e-2 * self.Khut * ((1.0 + self.Fw / self.Bh) ** (-0.25)) * (max_takeoff_weight ** 0.639) * (
                self.Nz ** 0.1) * (self.Sht ** 0.75) / self.Ltail * (self.Ky ** 0.7404) / cosT * (self.ARh ** 0.166) * (
                       (1.0 + self.Se / self.Sht) ** 0.1)

        weight_results[self.module_index] = W_ht

        return weight_results


# Vertical Wing
class VerticalWing(InitAirParams):

    def __init__(self, aircraft_name, aircraft_type, init_mission_class, engine_amplitude):
        self.aircraft_name = aircraft_name
        self.aircraft_type = aircraft_type

        self.module_index = 3

        super().__init__(self.aircraft_name, init_mission_class, engine_amplitude)

        # module name
        self.name = self.module_dict[self.module_index]

    def __call__(self, weight_results, other_args):
        """

        :param weight_results: ndarray
        :param other_args: [max_takeoff_weight]
        :return: ndarray
        """

        # max takeoff weight
        max_takeoff_weight = other_args[0]
        # from kg to lb
        max_takeoff_weight *= self.kg_to_lb

        # retreating angle
        cosV = np.cos(self.theta_v * np.pi / 180.0)

        W_vt = 2.6e-3 * ((1.0 + self.Hthv) ** 0.225) * (max_takeoff_weight ** 0.556) * \
               (self.Nz ** 0.536) / (self.Ltail ** 0.5) * (self.Svt ** 0.5) * (self.Kz ** 0.875) / cosV * (
                       self.ARv ** 0.35) / (self.tcroot ** 0.5)

        weight_results[self.module_index] = W_vt

        return weight_results


# Fuselage
class Fuselage(InitAirParams):

    def __init__(self, aircraft_name, aircraft_type, init_mission_class, engine_amplitude):
        self.aircraft_name = aircraft_name
        self.aircraft_type = aircraft_type
        self.module_index = 4

        super().__init__(self.aircraft_name, init_mission_class, engine_amplitude)

        # module name
        self.name = self.module_dict[self.module_index]

        self.structure = False

    # set Blended Wing Body configuration
    def set_bwb_config(self, blended_wing_body_class):

        # length of fuselage
        self.lf = blended_wing_body_class.overall_chord
        # surface area of fuselage
        self.Sf = blended_wing_body_class.fuselage_surface_area
        # fuselage wide length
        self.df = blended_wing_body_class.bwb_df

        self.structure = True
        self.blended_wing_body_fuselage_class = BlendedWingBodyWeight(blended_wing_body_class)

    def __call__(self, weight_results, other_args):
        """

        :param weight_results: ndarray
        :param other_args: [max_takeoff_weight]
        :return: ndarray
        """

        if not self.structure:
            # max takeoff weight
            max_takeoff_weight = other_args[0]
            # from kg to lb
            max_takeoff_weight *= self.kg_to_lb

            # calculate weight of fuselage
            W_fl = 0.328 * self.Kdoor * self.Klg * ((max_takeoff_weight * self.Nz) ** 0.5) * \
                   (self.lf ** 0.25) * (self.Sf ** 0.302) * ((1.0 + self.Kws) ** 0.04) * ((self.lf / self.df) ** 0.1)

        else:
            self.blended_wing_body_fuselage_class.run_fuselage()
            W_fl = self.blended_wing_body_fuselage_class.fuselage_weight

        # self.structure = False

        weight_results[self.module_index] = W_fl

        return weight_results


# Main Landing Gear
class MainLandingGear(InitAirParams):

    def __init__(self, aircraft_name, aircraft_type, init_mission_class, engine_amplitude):
        self.aircraft_name = aircraft_name
        self.aircraft_type = aircraft_type

        self.module_index = 5

        super().__init__(self.aircraft_name, init_mission_class,engine_amplitude)

        # module name
        self.name = self.module_dict[self.module_index]

    def __call__(self, weight_results, other_args):
        """

        :param weight_results: ndarray
        :param other_args: [max_takeoff_weight]
        :return: ndarray
        """

        # max_takeoff_weight
        # max_takeoff_weight = other_args

        # calculate weight of main landing gear
        W_mlg = 1.06e-2 * self.Kmp * (self.Wl ** 0.888) * (self.Nl ** 0.25) * (self.Lm ** 0.4) * \
                (self.Nmw ** 0.321) / (self.Nmss ** 0.5) * (self.Vstall ** 0.1)

        weight_results[self.module_index] = W_mlg

        return weight_results


# Nose Landing Gear
class NoseLandingGear(InitAirParams):

    def __init__(self, aircraft_name, aircraft_type, init_mission_class, engine_amplitude):
        self.aircraft_name = aircraft_name
        self.aircraft_type = aircraft_type

        self.module_index = 6

        super().__init__(self.aircraft_name, init_mission_class, engine_amplitude)

        # module name
        self.name = self.module_dict[self.module_index]

    def __call__(self, weight_results, other_args):
        """

        :param weight_results: ndarray
        :param other_args: [max_takeoff_weight]
        :return: ndarray
        """

        # calculate weight of nose landing gear
        W_nlg = 3.2e-2 * self.Knp * (self.Wl ** 0.646) * (self.Nl ** 0.2) * (self.Ln ** 0.5) * (self.Nnw ** 0.45)

        weight_results[self.module_index] = W_nlg

        return weight_results


# Nacelle
class Nacelle(InitAirParams):

    def __init__(self, aircraft_name, aircraft_type, init_mission_class, engine_amplitude):
        self.aircraft_name = aircraft_name
        self.aircraft_type = aircraft_type

        self.module_index = 7

        super().__init__(self.aircraft_name, init_mission_class, engine_amplitude)

        # module name
        self.name = self.module_dict[self.module_index]

    def __call__(self, weight_results, other_args):
        """

        :param weight_results: ndarray
        :param other_args: [engine_weight, fan_diameter, engine_length]
        :return: ndarray
        """

        # set other arguments
        engine_weight, fan_diameter, engine_length = other_args
        # unit change
        engine_weight = engine_weight * self.kg_to_lb
        fan_diameter = fan_diameter * self.m_to_ft
        engine_length = engine_length * self.m_to_ft
        W_ec = 2.331 * (engine_weight ** 0.901) * 1.18
        # nacelle area
        Sn = np.pi * fan_diameter * engine_length

        # calculate nacelle weight
        W_nac = 0.6724 * self.Kng * (self.Nlt ** 0.1) * (self.Nw ** 0.294) * (self.Nz ** 0.119) * \
                (W_ec ** 0.611) * (self.Nen ** 0.984) * (Sn ** 0.224)

        weight_results[self.module_index] = W_nac

        return weight_results


# Engine Control
class EngineControl(InitAirParams):

    def __init__(self, aircraft_name, aircraft_type, init_mission_class, engine_amplitude):
        self.aircraft_name = aircraft_name
        self.aircraft_type = aircraft_type

        self.module_index = 8

        super().__init__(self.aircraft_name, init_mission_class, engine_amplitude)

        # module name
        self.name = self.module_dict[self.module_index]

    def __call__(self, weight_results, other_args):
        """

        :param weight_results: ndarray
        :param other_args: [max_takeoff_weight]
        :return: ndarray
        """

        W_econ = 5.0 * self.Nen + 0.8 * self.Lec

        weight_results[self.module_index] = W_econ

        return weight_results


# Starter
class Starter(InitAirParams):

    def __init__(self, aircraft_name, aircraft_type, init_mission_class, engine_amplitude):
        self.aircraft_name = aircraft_name
        self.aircraft_type = aircraft_type

        self.module_index = 9

        super().__init__(self.aircraft_name, init_mission_class, engine_amplitude)

        # module name
        self.name = self.module_dict[self.module_index]

    def __call__(self, weight_results, other_args):
        """

        :param weight_results: ndarray
        :param other_args: [engine_weight, fan_diameter, engine_length]
        :return: ndarray
        """
        engine_weight, _, _ = other_args

        engine_weight *= self.kg_to_lb

        W_start = 49.19 * (engine_weight * self.Nen / 1.0e+3) ** 0.541

        weight_results[self.module_index] = W_start

        return weight_results


# Fuel System
class FuelSystem(InitAirParams):

    def __init__(self, aircraft_name, aircraft_type, init_mission_class, engine_amplitude):
        self.aircraft_name = aircraft_name
        self.aircraft_type = aircraft_type

        self.module_index = 10

        super().__init__(self.aircraft_name, init_mission_class, engine_amplitude)

        # module name
        self.name = self.module_dict[self.module_index]

    def __call__(self, weight_results, other_args):
        """

        :param weight_results: ndarray
        :param other_args: [max_takeoff_weight]
        :return: ndarray
        """

        W_fsys = 1000.0

        weight_results[self.module_index] = W_fsys

        return weight_results


# Flight Control
class FlightControl(InitAirParams):

    def __init__(self, aircraft_name, aircraft_type, init_mission_class, engine_amplitude):
        self.aircraft_name = aircraft_name
        self.aircraft_type = aircraft_type

        self.module_index = 11

        super().__init__(self.aircraft_name, init_mission_class, engine_amplitude)

        # module name
        self.name = self.module_dict[self.module_index]

    def __call__(self, weight_results, other_args):
        """

        :param weight_results: ndarray
        :param other_args: [max_takeoff_weight]
        :return: ndarray
        """

        # max takeoff weight
        max_takeoff_weight = other_args[0]
        # from kg to lb
        max_takeoff_weight *= self.kg_to_lb

        # total area of control surface
        Scs = self.Scsw + self.Sht * 0.35 + self.Svt * 0.35

        Iyaw = ((self.BW + self.lf) ** 2.0) / 16.0 * max_takeoff_weight * (0.49 ** 2.0)

        W_fcon = 145.9 * (self.Nf ** 0.554) / (1.0 + self.Nm / self.Nf) * (Scs ** 0.2) * (Iyaw * 1.0e-6) ** 0.07

        weight_results[self.module_index] = W_fcon

        return weight_results


# APU
class APU(InitAirParams):

    def __init__(self, aircraft_name, aircraft_type, init_mission_class, engine_amplitude):
        self.aircraft_name = aircraft_name
        self.aircraft_type = aircraft_type

        self.module_index = 12

        super().__init__(self.aircraft_name, init_mission_class, engine_amplitude)

        # module name
        self.name = self.module_dict[self.module_index]

    def __call__(self, weight_results, other_args):
        """

        :param weight_results: ndarray
        :param other_args: [max_takeoff_weight]
        :return: ndarray
        """

        W_apu = 836.0

        weight_results[self.module_index] = W_apu

        return weight_results


# Instrument
class Instrument(InitAirParams):

    def __init__(self, aircraft_name, aircraft_type, init_mission_class, engine_amplitude):
        self.aircraft_name = aircraft_name
        self.aircraft_type = aircraft_type

        self.module_index = 13

        super().__init__(self.aircraft_name, init_mission_class, engine_amplitude)

        # module name
        self.name = self.module_dict[self.module_index]

    def __call__(self, weight_results, other_args):
        """

        :param weight_results: ndarray
        :param other_args: [max_takeoff_weight]
        :return: ndarray
        """

        W_inst = 4.509 * self.Kr * self.Ktp * (self.Nc ** 0.541) * self.Nen * np.sqrt(self.BW + self.lf)

        weight_results[self.module_index] = W_inst

        return weight_results


# Hydraulics
class Hydraulics(InitAirParams):

    def __init__(self, aircraft_name, aircraft_type, init_mission_class, engine_amplitude):
        self.aircraft_name = aircraft_name
        self.aircraft_type = aircraft_type

        self.module_index = 14

        super().__init__(self.aircraft_name, init_mission_class, engine_amplitude)

        # module name
        self.name = self.module_dict[self.module_index]

    def __call__(self, weight_results, other_args):
        """

        :param weight_results: ndarray
        :param other_args: [max_takeoff_weight]
        :return: ndarray
        """

        W_hyd = 0.2673 * self.Nf * ((self.BW + self.lf) ** 0.937)

        weight_results[self.module_index] = W_hyd

        return weight_results


# Electric
class Electric(InitAirParams):

    def __init__(self, aircraft_name, aircraft_type,init_mission_class, engine_amplitude):
        self.aircraft_name = aircraft_name
        self.aircraft_type = aircraft_type

        self.module_index = 15

        super().__init__(self.aircraft_name, init_mission_class, engine_amplitude)

        # module name
        self.name = self.module_dict[self.module_index]

    def __call__(self, weight_results, other_args):
        """

        :param weight_results: ndarray
        :param other_args: [max_takeoff_weight]
        :return: ndarray
        """

        W_hyd = 1755.0

        weight_results[self.module_index] = W_hyd

        return weight_results


# Avionics
class Avionics(InitAirParams):

    def __init__(self, aircraft_name, aircraft_type, init_mission_class, engine_amplitude):
        self.aircraft_name = aircraft_name
        self.aircraft_type = aircraft_type

        self.module_index = 16

        super().__init__(self.aircraft_name, init_mission_class, engine_amplitude)

        # module name
        self.name = self.module_dict[self.module_index]

    def __call__(self, weight_results, other_args):
        """

        :param weight_results: ndarray
        :param other_args: [max_takeoff_weight]
        :return: ndarray
        """

        # UAV
        # W_uav = 1100
        # W_avi = 1.73 * W_uav ** 0.983

        W_avi = 2141.0

        weight_results[self.module_index] = W_avi

        return weight_results


# Furnishing
class Furnishing(InitAirParams):

    def __init__(self, aircraft_name, aircraft_type, init_mission_class, engine_amplitude):
        self.aircraft_name = aircraft_name
        self.aircraft_type = aircraft_type

        self.module_index = 17

        super().__init__(self.aircraft_name, init_mission_class, engine_amplitude)

        # module name
        self.name = self.module_dict[self.module_index]

    def __call__(self, weight_results, other_args):
        """

        :param weight_results: ndarray
        :param other_args: [max_takeoff_weight]
        :return: ndarray
        """

        Wc = 41079.0
        W_fur = 0.0577 * (self.Nc ** 0.1) * (Wc ** 0.393) * (self.Sf ** 0.75)

        weight_results[self.module_index] = W_fur

        return weight_results


# AirConditioner
class AirConditioner(InitAirParams):

    def __init__(self, aircraft_name, aircraft_type, init_mission_class, engine_amplitude):
        self.aircraft_name = aircraft_name
        self.aircraft_type = aircraft_type

        self.module_index = 18

        super().__init__(self.aircraft_name, init_mission_class, engine_amplitude)

        # module name
        self.name = self.module_dict[self.module_index]

    def __call__(self, weight_results, other_args):
        """

        :param weight_results: ndarray
        :param other_args: [max_takeoff_weight]
        :return: ndarray
        """

        W_aircon = 1274.0

        weight_results[self.module_index] = W_aircon

        return weight_results


# Anti - ice
class AntiIce(InitAirParams):

    def __init__(self, aircraft_name, aircraft_type, init_mission_class, engine_amplitude):
        self.aircraft_name = aircraft_name
        self.aircraft_type = aircraft_type

        self.module_index = 19

        super().__init__(self.aircraft_name, init_mission_class, engine_amplitude)

        # module name
        self.name = self.module_dict[self.module_index]

    def __call__(self, weight_results, other_args):
        """

        :param weight_results: ndarray
        :param other_args: [max_takeoff_weight]
        :return: ndarray
        """

        # max takeoff weight
        max_takeoff_weight = other_args[0]
        # from kg to lb
        max_takeoff_weight *= self.kg_to_lb

        W_anti = 0.002 * max_takeoff_weight

        weight_results[self.module_index] = W_anti

        return weight_results


# Handling gear
class HandlingGear(InitAirParams):

    def __init__(self, aircraft_name, aircraft_type, init_mission_class, engine_amplitude):
        self.aircraft_name = aircraft_name
        self.aircraft_type = aircraft_type

        self.module_index = 20

        super().__init__(self.aircraft_name, init_mission_class, engine_amplitude)

        # module name
        self.name = self.module_dict[self.module_index]

    def __call__(self, weight_results, other_args):
        """

        :param weight_results: ndarray
        :param other_args: [max_takeoff_weight]
        :return: ndarray
        """

        # max takeoff weight
        max_takeoff_weight = other_args[0]
        # from kg to lb
        max_takeoff_weight *= self.kg_to_lb

        W_hand = 3.0e-4 * max_takeoff_weight

        weight_results[self.module_index] = W_hand

        return weight_results


# Passenger equipment
class PassengerEquip(InitAirParams):

    def __init__(self, aircraft_name, aircraft_type, init_mission_class, engine_amplitude):
        self.aircraft_name = aircraft_name
        self.aircraft_type = aircraft_type

        self.module_index = 22

        super().__init__(self.aircraft_name, init_mission_class, engine_amplitude)

        # module name
        self.name = self.module_dict[self.module_index]

    def __call__(self, weight_results, other_args):
        """

        :param weight_results: ndarray
        :param other_args: [max_takeoff_weight]
        :return: ndarray
        """

        W_pe = 9597.0

        weight_results[self.module_index] = W_pe

        return weight_results
