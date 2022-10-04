from standard_air_utils import StandardAir
import numpy as np
import json


class InitPhysicCondition(object):

    def __init__(self, altitude, mach):

        # define class for calculating physical parameters
        sa = StandardAir(altitude)

        # Static temperature and Static Pressure
        self.static_T, self.static_P = sa.T, sa.P

        # Heat Constant(Cp)
        # before Combustion Chamber:1004,after Compressor:1156
        self.cp_comp_before = 1004.0
        self.cp_comp_after = 1156.0

        # gamma
        self.gamma_comp_before = 1.4  # 単原子分子
        self.gamma_comp_after = 1.33  # 二原子分子

        # Air Constant(R)
        self.R = self.cp_comp_before / (self.gamma_comp_before / (self.gamma_comp_before - 1))

        # Cruise Velocity(m/s)
        self.V_jet = 0

        if altitude > 0:
            self.V_jet = mach * np.sqrt(self.gamma_comp_before * self.R * self.static_T)

        # print(self.aircraft_data['Altitude'])

        # off design point params
        self.off_params = [0.0, 0.0, 1.0, 0.0, -0.003, 1.0, 1.0, 1.0, 1.0, 1.0, 0]  # default

    # for calculating design off point thermal analysis
    def set_design_off_point(self, off_param_args):
        off_altitude, off_mach, off_required_thrust = off_param_args

        # set altitude and mach number

        self.off_params[0] = off_altitude

        self.off_params[1] = off_mach

        # set required thrust at ground
        self.off_params[-1] = off_required_thrust


def test_initp():
    altitude = 10668
    mach = 0.78

    pc = InitPhysicCondition(altitude, mach)

    print('Jet Velocity [m/s]:', pc.V_jet)


if __name__ == '__main__':
    test_initp()
