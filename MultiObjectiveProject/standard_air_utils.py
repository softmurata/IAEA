import numpy as np


class StandardAir(object):

    def __init__(self, Z):

        # Initialize values at 0 meter point
        self.T0 = 15  # temperature
        self.P0 = 101325  # Pa Pressure
        self.rou0 = 1.225  # kg/m^3　density
        self.a0 = 340.29  # m/s　tone velocity
        self.mu0 = 1.7849 * 1e-5  # Ns/m^2 viscous ratio
        self.kai0 = 1.4607 * 1e-5  # m^2/a dynamic viscous ratio

        self.kelvin_ch = 273.15

        # given variables
        self.Z = Z  # height

        if self.Z > 1000:
            self.Z /= 1e+3

        # calculate geo potential
        self.geopotential()
        # calculate static temperature and pressure
        self.temperature_and_pressure()
        # calculate density
        self.density()
        # calculate tone velocity
        self.tone_velocity()
        # calculate viscous
        self.viscousity_coef()
        # dynamic viscous
        self.dynamic_viscous_coef()
        self.P *= 0.99735
        self.T *= 0.9995
        # print('Pressure[pa]:',self.P,'temperature[K]:',self.T,'density[kg/m^3]:',self.rou,'tone_velocity[m/s]:',self.a,'vicous_density[Ns/m^2]:',self.mu,'kai:',self.kai)

    # Geo potential
    def geopotential(self):
        """
        calculate geo potential
        """
        r0 = 6356.766  # [km]
        self.H = r0 * self.Z / (r0 + self.Z)  # km

    # 温度計算
    def temperature_and_pressure(self):
        """
        calculate static temperature and pressure according to standard air
        """

        if 0 <= self.H <= 11:
            self.T = self.T0 - 6.5 * self.H
            self.P = self.P0 * (288.15 / (self.T + self.kelvin_ch)) ** (-5.256)

        elif 11.0 < self.H <= 20.0:
            self.T = -56.5
            self.P = 22632.064 * np.exp(-0.1577 * (self.H - 11))

        elif 20.0 < self.H <= 32:
            self.T = -76.5 + self.H
            self.P = 5474.889 * (216.65 / (self.T + self.kelvin_ch)) ** (34.163)

        elif 32 < self.H <= 47:
            self.T = -134.1 + 2.8 * self.H
            self.P = 868.019 * (228.65 / (self.T + self.kelvin_ch)) ** (12.201)

        elif 47 < self.H <= 51:
            self.T = -2.5
            self.P = 110.906 * np.exp(-0.1262 * (self.H - 47))

        elif 51 < self.H <= 71:
            self.T = 140.3 - 2.8 * self.H
            self.P = 66.939 * (270.65 / (self.T + self.kelvin_ch)) ** (-12.201)

        elif 71 <= self.H <= 84.852:
            self.T = 83.5 - 2.0 * self.H
            self.P = 3.956 * (214.65 / (self.T + self.kelvin_ch)) ** (-17.082)

        self.T += self.kelvin_ch  # K

    # 密度計算
    def density(self):
        """
        calculate density
        """

        self.rou = 0.0034837 * self.P / (self.T)

    # 音速
    def tone_velocity(self):
        """
        calculate tone velocity
        """

        self.a = 20.0468 * np.sqrt(self.T)

    # 粘性係数
    def viscousity_coef(self):
        """
        calculate viscous coefficient
        """

        S = 110.4  # サザーランド定数
        beta = 1.458e-6  # 係数
        self.mu = beta * (self.T) ** 1.5 / (self.T + S)

    # 動粘性係数
    def dynamic_viscous_coef(self):
        """
        calculate dynamic viscous coefficient
        """

        self.kai = self.mu / self.rou


if __name__ == '__main__':
    Z = 10668  # [m]
    sa = StandardAir(Z)
