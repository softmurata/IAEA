import numpy as np
from air_shape import BlendedWingBodyShape
from mission_tuning import InitMission
from design_variable import DesignVariable
from thermal_dp import CalcDesignPoint
from thermal_doff import CalcDesignOffPoint
from engine_weight import EngineWeight


class BlendedWingBodyWeight(object):

    def __init__(self, air_shape_class):

        self.air_shape_class = air_shape_class

        self.diff_num = len(self.air_shape_class.mass_outside_shape)

        self.max_x_length = self.air_shape_class.mass_inside_shape[-1][0]

        self.y_coords = self.air_shape_class.mass_inside_shape

        self.inside_coords = []

        self.delta_x = self.max_x_length / self.diff_num

        # required variables
        self.secure_coef = 1.5  # value of secure modulus

        # difference between outer and insider pressure
        self.pressure_diff = 80  # [kPa] => 10 ** 3 [N/m**2]
        # stress reaching to fatigue collapse
        self.sigma_fatigue = 135  # [N/mm ** 2]
        # density of inside shelter
        self.density_core = 50  # [kg/m ** 3]
        # density of surface
        self.density_skin = 2770  # [kg/m ** 3]

        self.ft_to_m = 0.3048

        # mass data
        self.mass_fracs_skin = []
        self.mass_fracs_wall = []

        # total weight
        self.fuselage_weight = 0

    def create_inside_coords(self):
        x = 0
        y = 0
        for _ in range(self.diff_num):
            if 0 <= x <= self.y_coords[1][0]:
                slope = (self.y_coords[1][1] - self.y_coords[0][1]) / (self.y_coords[0][0] - 0)
                y = slope * x

            elif self.y_coords[1][0] <= x <= self.y_coords[2][0]:
                slope = (self.y_coords[2][1] - self.y_coords[1][1]) / (self.y_coords[2][0] - self.y_coords[1][0])
                y = slope * (x - self.y_coords[1][0]) + self.y_coords[1][1]

            elif self.y_coords[2][0] <= x <= self.y_coords[3][0]:
                slope = (self.y_coords[3][1] - self.y_coords[2][1]) / (self.y_coords[3][0] - self.y_coords[2][0])
                y = slope * (x - self.y_coords[2][0]) + self.y_coords[2][1]

            elif self.y_coords[3][0] <= x <= self.y_coords[4][0]:
                slope = (self.y_coords[4][1] - self.y_coords[3][1]) / (self.y_coords[4][0] - self.y_coords[3][0])
                y = slope * (x - self.y_coords[3][0]) + self.y_coords[3][1]

            elif self.y_coords[4][0] <= x:
                slope = (self.y_coords[5][1] - self.y_coords[4][1]) / (self.y_coords[5][0] - self.y_coords[4][0])
                y = slope * (x - self.y_coords[4][0]) + self.y_coords[4][1]

            self.inside_coords.append([x, y])
            x += self.delta_x

    def calc_eclipse_length(self, a, b):

        r = (a - b) / (a + b)

        if np.isnan(r):

            r = 0

        return np.pi * (a + b) * (1.0 + 3 * r ** 2 / (10 + np.sqrt(4 - 3 * r ** 2)))

    def calculate_fuselage(self):
        delta_l = self.air_shape_class.overall_chord / self.diff_num
        maxtskin = 0
        for idx in range(self.diff_num):
            x, w = self.inside_coords[idx]
            h = self.air_shape_class.cabin_height
            a, bu, bl = self.air_shape_class.mass_outside_shape[idx]

            w = w * self.ft_to_m
            h = h * self.ft_to_m
            a = a * self.ft_to_m
            bu = bu * self.ft_to_m
            bl = bl * self.ft_to_m

            cross_angle = np.arctan(h / w)
            around_length = 0.5 * self.calc_eclipse_length(a, bu) + 0.5 * self.calc_eclipse_length(a, bl)
            R = around_length / (2 * np.pi)
            # skin thickness
            tskin = self.secure_coef * self.pressure_diff * R / self.sigma_fatigue * 1e-3

            mskin = np.pi * self.density_skin * (tskin * (a + min(bu, bl)) + tskin ** 2) * delta_l

            R1 = around_length * (4 * cross_angle / (2 * np.pi))
            R2 = around_length * ((2.0 * np.pi - 4 * cross_angle) / (2.0 * np.pi)) * 3 / 4
            # vertical thickness
            Fres = self.secure_coef * self.pressure_diff * delta_l * abs(R2 - R1)  # abs((min(bl, bu) - a))
            # Vertical and Horizontal Force
            Fv = Fres * np.cos(cross_angle)
            Fh = Fres * np.sin(cross_angle)

            tvert = Fv / (self.sigma_fatigue * delta_l) * 1e-3
            thori = Fh / (self.sigma_fatigue * delta_l) * 1e-3

            mwall = self.density_core * (2 * tvert + 2 * thori) * delta_l

            self.fuselage_weight += (mskin + mwall)
            self.mass_fracs_skin.append(mskin)
            self.mass_fracs_wall.append(mwall)

            if tskin > maxtskin:
                maxtskin = tskin

            # print('')
            # print('step {}:'.format(idx))
            # print('tskin:', tskin, 'tvert:', tvert, 'thori:', thori)
            # print('cross angle:', cross_angle * 180.0 / np.pi)
            # print('mskin:', mskin, 'mwall:', mwall)
            # print('')

        print('max thickness of skin:', maxtskin)

    def run_fuselage(self):
        self.create_inside_coords()
        self.calculate_fuselage()



def test():
    # global variables
    aircraft_name = 'A320'
    aircraft_type = 'BWB'
    engine_name = 'V2500'
    propulsion_type = 'turbofan'

    # data path
    aircraft_data_path = './DataBase/aircraft_test.json'
    engine_data_path = './DataBase/engine_test.json'
    mission_data_path = './Missions/fuelburn18000.json'

    data_base_args = [aircraft_data_path, engine_data_path, mission_data_path]

    # build init mission class
    init_mission_class = InitMission(aircraft_name, engine_name, aircraft_data_path, engine_data_path)

    # design variable class
    dv = DesignVariable(propulsion_type, aircraft_type)
    dv_list = [4.7, 30.0, 1.61, 1380, 1.0, 0.8, 0.48, 1.0, 0.2]
    # thermal design variable class
    thermal_design_variables = dv.set_design_variable(dv_list)

    print(thermal_design_variables[-5:])

    # design point params
    design_point_params = [10668, 0.78]

    # off design point params
    off_altitude = 0
    off_mach = 0
    off_required_thrust = 133000
    off_param_args = [off_altitude, off_mach, off_required_thrust]

    # build calc design point class
    calc_design_point_class = CalcDesignPoint(aircraft_name, engine_name, aircraft_type, propulsion_type, thermal_design_variables, init_mission_class, design_point_params, data_base_args)

    calc_design_point_class.run_dp()

    calc_design_point_class.objective_func_dp()

    # rev args
    rev_args = [1.105, 1.0]

    # build calc off design point class
    calc_off_design_point_class = CalcDesignOffPoint(aircraft_name, engine_name, aircraft_type, propulsion_type, thermal_design_variables, off_param_args, design_point_params, data_base_args, calc_design_point_class)

    calc_off_design_point_class.run_off(rev_args)

    calc_off_design_point_class.objective_func_doff()


    # build engine weight class
    engine_weight_class = EngineWeight(aircraft_name, engine_name, aircraft_type, propulsion_type, engine_data_path, calc_design_point_class, calc_off_design_point_class)

    engine_weight_class.run_engine()

    init_shape_params = 0.8

    # engine amplitude
    init_mission_class.load_mission_config(mission_data_path)
    baseline_engine_weight = init_mission_class.engine_weight
    engine_amplitude = engine_weight_class.total_engine_weight / baseline_engine_weight
    other_args = engine_amplitude

    bwb = BlendedWingBodyShape(aircraft_name, init_mission_class, thermal_design_variables, engine_weight_class, init_shape_params, data_base_args, other_args)

    bwb.run_airshape(drawing=False)

    bwbw = BlendedWingBodyWeight(bwb)

    bwbw.run_fuselage()

    print(bwbw.fuselage_weight)


if __name__ == '__main__':
    test()


