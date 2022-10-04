import numpy as np
import json
import time
from init_airparams import InitAirParams
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from various_curve import *
from air_shape_utils import *

# for cython
from mission_tuning_cy import InitMission
from design_variable_cy import DesignVariable
from thermal_doff_cy import calc_off_design_point
from thermal_dp_cy import calc_design_point
from engine_weight_cy import calc_engine_weight


# According shape of aircraft, each shape have class
class NormalShape(InitAirParams):
    """
    Attributes
    -------------
    name: str

    engine_weight_class: class object

    """

    def __init__(self, aircraft_name, init_mission_class, engine_weight_class, engine_mounting_positions, other_args, distfan_mount='upper'):

        engine_amplitude = other_args

        self.name = 'normal'

        super().__init__(aircraft_name, init_mission_class, engine_amplitude)

        # load aircraft data
        self.set_main_config()

        # set engine weight class (Already calculate)
        self.engine_weight_class = engine_weight_class

        # set engine mounting position coefficients
        self.engine_mounting_coefx, self.engine_mounting_coefy = engine_mounting_positions

        # design variables for aircraft shape
        self.theta_front = 20
        self.theta_back = 20

        # Trailing shape coefficient
        self.ts_coef = 1.4

        # unit change
        self.kg_to_lb = 2.20462
        self.m_to_ft = 3.2804

        # Initialize wing configuration
        # Main Wing
        self.main_croot = self.Swref / (self.BW * (1.0 + self.Lambda))
        self.main_ctip = self.Lambda * self.main_croot
        # Vertical Wing
        self.BV = np.sqrt(self.ARv * self.Svt)
        self.vert_croot = self.Svt / (self.BV * (1.0 + self.Lambda))
        self.vert_ctip = self.Lambda * self.vert_croot
        # Horizontal Wing
        self.hori_croot = self.Sht / ((self.Bh - self.df * 0.3) * (1.0 + self.Lambda))
        self.hori_ctip = self.Lambda * self.hori_croot

        # Initialize values
        self.fuselage_length = self.lf  # Total fuselage length [ft]
        self.fuselage_volume = None  # Total fuselage volume  [ft**3]
        self.fuselage_surface_area = 0  # Total fuselage surface area [ft**2]

        # Wet area
        self.main_wing_wet_area = None  # wet area of main wing
        self.hori_wing_wet_area = None  # wet area of horizontal wing
        self.vert_wing_wet_area = None  # wet area of vertical wing
        self.fuselage_wet_area = None

        self.distfan_mount = distfan_mount  # place for setting distributed fan

        self.upper_distributed_fan_dists = []
        self.lower_distirbuted_fan_dists = []

        self.upper_shelter_dists = []
        self.lower_shelter_dists = []

    # compose cross area distribution
    def compose_cross_area_dist(self):

        args = [self.lf, self.df, self.ts_coef, self.theta_front, self.theta_back, self.fuselage_surface_area]
        self.upper_cross_area_dist, self.lower_cross_area_dist, self.fuselage_surface_area = cross_area(args)

    # helper function for function called 'define_wing_config'
    def hori_wing_dist(self, upper_wing_area_dists, lower_wing_area_dists, wing_mounting_position, retreating_angle, wing_width, y_init,
                  airfoil_args):
        """

        :param upper_wing_area_dists: Upper wing distribution
        :param lower_wing_area_dists: Lower wing distribution
        :param wing_mounting_position:
        :param retreating_angle: theta
        :param wing_width: BW, BH, BV
        :param y_init:
        :param airfoil_args: [croot, ctip, troot, ttip]
        :return:
        """

        args = [upper_wing_area_dists, lower_wing_area_dists, wing_mounting_position, retreating_angle, wing_width, y_init, airfoil_args]
        upper_wing_area_dists, lower_wing_area_dists = hori_wing(args)

        return upper_wing_area_dists, lower_wing_area_dists

    # helper function for function called 'define_wing_config'
    def vert_wing_dist(self, upper_wing_area_dists, wing_mounting_position, retreating_angle, wing_width, z_init, airfoil_args):
        """

        :param upper_wing_area_dists:
        :param wing_mounting_position:
        :param retreating_angle:
        :param wing_width:
        :param z_init:
        :param airfoil_args:
        :return:
        """

        args = [upper_wing_area_dists, wing_mounting_position, retreating_angle, wing_width, z_init, airfoil_args]
        upper_wing_area_dists = vert_wing(args)

        return upper_wing_area_dists

    # define wing configurations (main wing, horizontal wing, vertical wing)
    def define_wing_config(self):
        # wing mounting point
        self.main_wing_mounting_point = self.lf / 3
        self.tail_wing_mounting_point = self.lf * 0.95

        # thickness of wing
        # Main Wing
        self.main_troot = self.main_croot * self.tcroot
        self.main_ttip = self.main_troot * 0.7
        # Vertical Wing
        self.vert_troot = self.vert_croot * self.tcroot
        self.vert_ttip = self.vert_troot * 0.7
        # Horizontal Wing
        self.hori_troot = self.hori_croot * self.tcroot
        self.hori_ttip = self.hori_troot * 0.7

        # Main Wing
        self.main_upper_wing_area_dists = []
        self.main_lower_wing_area_dists = []
        main_y_init = self.df * 0.5
        main_airfoil_args = [self.main_croot, self.main_ctip, self.main_troot, self.main_ttip]

        self.main_upper_wing_area_dists, self.main_lower_wing_area_dists = self.hori_wing_dist(self.main_upper_wing_area_dists, self.main_lower_wing_area_dists, self.main_wing_mounting_point, self.theta, self.BW * 0.5, main_y_init, main_airfoil_args)

        # Horizontal Wing
        self.hori_upper_wing_area_dists = []
        self.hori_lower_wing_area_dists = []
        hori_y_init = self.df * 0.1
        hori_airfoil_args = [self.hori_croot, self.hori_ctip, self.hori_troot, self.hori_ttip]

        self.hori_upper_wing_area_dists, self.hori_lower_wing_area_dists = self.hori_wing_dist(self.hori_upper_wing_area_dists, self.hori_lower_wing_area_dists, self.tail_wing_mounting_point, self.theta_h, self.Bh * 0.5, hori_y_init, hori_airfoil_args)

        # Vertical Wing
        self.vert_upper_wing_area_dists = []
        self.vert_lower_wing_area_dists = []
        vert_z_init = self.df * 0.5 * self.ts_coef
        vert_airfoil_args = [self.vert_croot, self.vert_ctip, self.vert_troot, self.vert_ttip]

        self.vert_upper_wing_area_dists = self.vert_wing_dist(self.vert_upper_wing_area_dists, self.tail_wing_mounting_point, self.theta_v, self.BV, vert_z_init, vert_airfoil_args)

    # determine engine shape configuration
    def define_engine_config(self):
        args = [self.engine_mounting_coefx, self.engine_mounting_coefy, self.BW, self.df, self.main_croot, self.main_troot, self.main_wing_mounting_point, self.engine_weight_class, self.m_to_ft]
        self.upper_engine_area_dists, self.lower_engine_area_dists, self.engine_mounting_position_x, self.engine_mounting_position_y = engine(args)


    def define_distributed_fan_config(self):

        args = [self.engine_mounting_coefx, self.engine_mounting_coefy, self.BW, self.df, self.main_croot, self.main_troot, self.theta, self.main_wing_mounting_point, self.engine_weight_class, self.m_to_ft, self.distfan_mount]

        self.upper_distributed_fan_dists, self.lower_distirbuted_fan_dists, self.distributed_fan_mounting_positons_x, self.distributed_fan_mounting_positons_y = distributed_fan(args)

    def define_shelter(self):

        args = [self.upper_distributed_fan_dists, self.df, self.main_troot, self.theta, self.main_wing_mounting_point, self.engine_weight_class, self.m_to_ft]
        self.upper_shelter_dists, self.lower_shelter_dists = shelter(args)

    # if you want to calculate the weight or aerodynamic performances more in detail, you should change
    # this function into another model
    # Now, front and back shape think of as the triangle pole , fuselage shape think of as circle pole


    def calc_volume(self):
        r_cross = 0.5 * self.df * self.ts_coef
        # section1 Volume
        V1 = (2 / 3) * np.pi * (r_cross ** 2 / self.ts_coef) * (r_cross / np.tan(self.theta_front * np.pi / 180.0))

        # section2 Volume
        V2 = np.pi * (r_cross ** 2 / self.ts_coef) * (self.lf - (r_cross / np.tan(self.theta_front * np.pi / 180.0)) - (2 * r_cross / np.tan(self.theta_back * np.pi / 180.0)))

        # section3 Volume
        V3 = (2 / 3) * np.pi * (r_cross ** 2 / self.ts_coef) * (2 * r_cross / np.tan(self.theta_back * np.pi / 180.0))

        self.fuselage_volume = V1 + V2 + V3

        # calculate top and side area for wet area
        L_dash = self.lf - (r_cross / np.tan(self.theta_front * np.pi / 180.0) + (2 * r_cross / np.tan(self.theta_back * np.pi / 180.0)))
        # Top
        s_top_fus = 0.5 * np.pi * (0.5 * self.df) * (r_cross / np.tan(self.theta_front * np.pi / 180.0)) + 0.5 * np.pi * (0.5 * self.df) * (2 * r_cross / np.tan(self.theta_back * np.pi / 180.0)) + self.df * L_dash
        s_top_mwing = 0.5 * self.main_croot * self.BW * (1.0 + self.Lambda)
        s_top_hwing = 0.5 * self.hori_croot * self.Bh * (1.0 + self.Lambda)

        self.main_wing_wet_area = s_top_mwing
        self.hori_wing_wet_area = s_top_hwing

        # Side
        s_side_fus = 0.5 * np.pi * r_cross ** 2 / np.tan(self.theta_front * np.pi / 180.0) + 0.25 * (np.pi * (2.0 * r_cross) ** 2 / np.tan(self.theta_back * np.pi / 180.0)) + 2.0 * r_cross * L_dash
        s_side_vwing = 0.5 * self.vert_croot * self.BV * (1.0 + self.Lambda)

        self.vert_wing_wet_area = s_side_vwing

        # calculate total wet area
        self.fuselage_wet_area = 1.7 * (s_top_fus + s_side_fus)

        self.wet_area = self.fuselage_wet_area

    # draw pictures
    def draw_pictures(self):
        # fuselage all data
        fuselage_upper_datas = np.array(self.upper_cross_area_dist)
        fuselage_lower_datas = np.array(self.lower_cross_area_dist)

        # Main wing all data
        main_upper_datas = np.array(self.main_upper_wing_area_dists)
        main_lower_datas = np.array(self.main_lower_wing_area_dists)

        # horizontal wing all data
        hori_upper_datas = np.array(self.hori_upper_wing_area_dists)
        hori_lower_datas = np.array(self.hori_lower_wing_area_dists)

        # vertical wing all data
        vert_upper_datas = np.array(self.vert_upper_wing_area_dists)

        # engine all data
        engine_upper_datas = np.array(self.upper_engine_area_dists)
        engine_lower_datas = np.array(self.lower_engine_area_dists)

        if len(self.upper_distributed_fan_dists) != 0:
            # distributed fan all data
            distributed_upper_data = np.array(self.upper_distributed_fan_dists)
            distributed_lower_data = np.array(self.lower_distirbuted_fan_dists)

            if self.distfan_mount == 'upper':
                # shelter
                shelter_upper_data = np.array(self.upper_shelter_dists)
                shelter_lower_data = np.array(self.lower_shelter_dists)


        dfu_x, dfu_y, dfu_z = fuselage_upper_datas[:, 0], fuselage_upper_datas[:, 1], fuselage_upper_datas[:, 2]
        dfl_x, dfl_y ,dfl_z = fuselage_lower_datas[:, 0], fuselage_lower_datas[:, 1], fuselage_lower_datas[:, 2]

        dmu_x, dmu_y, dmu_z = main_upper_datas[:, 0], main_upper_datas[:, 1], main_upper_datas[:, 2]
        dml_x, dml_y, dml_z = main_lower_datas[:, 0], main_lower_datas[:, 1], main_lower_datas[:, 2]

        dhu_x, dhu_y, dhu_z = hori_upper_datas[:, 0], hori_upper_datas[:, 1], hori_upper_datas[:, 2]
        dhl_x, dhl_y, dhl_z = hori_lower_datas[:, 0], hori_lower_datas[:, 1], hori_lower_datas[:, 2]

        dvu_x, dvu_y, dvu_z = vert_upper_datas[:, 0], vert_upper_datas[:, 1], vert_upper_datas[:, 2]

        deu_x, deu_y, deu_z = engine_upper_datas[:, 0], engine_upper_datas[:, 1], engine_upper_datas[:, 2]
        del_x, del_y, del_z = engine_lower_datas[:, 0], engine_lower_datas[:, 1], engine_lower_datas[:, 2]

        if len(self.upper_distributed_fan_dists) != 0:
            ddu_x, ddu_y, ddu_z = distributed_upper_data[:, 0], distributed_upper_data[:, 1], distributed_upper_data[:, 2]
            ddl_x, ddl_y, ddl_z = distributed_lower_data[:, 0], distributed_lower_data[:, 1], distributed_lower_data[:, 2]

            if self.distfan_mount == 'upper':
                dsu_x, dsu_y, dsu_z = shelter_upper_data[:, 0], shelter_upper_data[:, 1], shelter_upper_data[:, 2]
                dsl_x, dsl_y, dsl_z = shelter_lower_data[:, 0], shelter_lower_data[:, 1], shelter_lower_data[:, 2]

        fig = plt.figure(figsize=(10, 8))
        ax = Axes3D(fig)

        ax.scatter(dfu_x, dfu_y, dfu_z, c='b')
        ax.scatter(dfl_x, dfl_y, dfl_z, c='b')

        ax.scatter(dmu_x, dmu_y, dmu_z, c='r')
        ax.scatter(dml_x, dml_y, dml_z, c='r')

        ax.scatter(dhu_x, dhu_y, dhu_z, c='g')
        ax.scatter(dhl_x, dhl_y, dhl_z, c='g')

        ax.scatter(dvu_x, dvu_y, dvu_z)

        ax.scatter(deu_x, deu_y, deu_z)
        ax.scatter(del_x, del_y, del_z)

        if len(self.upper_distributed_fan_dists) != 0:
            ax.scatter(ddu_x, ddu_y, ddu_z)
            ax.scatter(ddl_x, ddl_y, ddl_z)

            if self.distfan_mount == 'upper':
                ax.scatter(dsu_x, dsu_y, dsu_z)
                ax.scatter(dsl_x, dsl_y, dsl_z)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([140, 0])
        ax.set_ylim([60, -60])
        ax.set_zlim([-60, 60])

        plt.show()

    # by matching the surface area, we have to change angles of leading edge and trailing edge
    # two split method
    # A320 tuning results: ts_coef:1.4589, theta_back: 10.116
    def tuning_surface_area(self, data_path):
        """

        :param data_path:
        :return:
        """

        # This method has two loop. One is the shape coefficient(ts_coef), the other is Trailing edge angle (theta_back)
        ts_coef_step = 0.1
        ts_count = 0
        # Residual for shape coefficient
        rests = 0.0
        restsold = 0.0

        while True:

            if ts_count == 300:
                break

            # Initialize theta back
            self.theta_back = 20
            theta_back_step = 0.1
            # Residual for Trailing edge of angle
            ressurfarea = 0.0
            ressurfareaold = 0.0

            # calculate count
            theta_count = 0

            while True:

                if theta_count == 300:
                    break

                # calculate the fuselage surface area
                self.fuselage_surface_area = 0
                self.compose_cross_area_dist()
                # residual of surface area
                ressurfarea = 1.0 - self.fuselage_surface_area / self.Sf
                # check convergence
                if ressurfarea * ressurfareaold < 0.0:
                    break

                # update theta_back
                self.theta_back += np.sign(ressurfarea) * theta_back_step

                ressurfareaold = ressurfarea

                theta_count += 1

            # near the solution, diminish the steps
            rests = ressurfarea

            if rests * restsold < 0.0:
               ts_coef_step *= 0.5

            # confirmation
            print('trailing shape coef:', self.ts_coef, 'residual surface area:', rests, 'angle of trailing edge:', self.theta_back)

            if abs(rests) < 1.0e-5:
                break

            self.ts_coef += np.sign(rests) * ts_coef_step

            restsold = rests

            ts_count += 1

        # Convergence Check
        # print('front_angle:', self.theta_front, 'back_angle:', self.theta_back)
        # print('fuselage surface area:', self.fuselage_surface_area, 'target:', self.Sf)
        # save tuning coefficients
        self.save_normal_aircraft_config(data_path)

    # save tuning coefficients
    def save_normal_aircraft_config(self, data_path):
        """

        :param data_path:
        :return:
        """
        # open the file as reading type
        f = open(data_path, 'r')
        aircraft_file = json.load(f)

        # new dictionary
        previous_dict = aircraft_file[self.aircraft_name]
        new_dict = {}
        # Initialize additional key and value
        new_key_name = ['theta_front', 'theta_back', 'ts_coef']
        new_vals = [self.theta_front, self.theta_back, self.ts_coef]

        # put into previous items
        for key, val in previous_dict.items():
            new_dict[key] = val

        # put into new items
        for key, val in zip(new_key_name, new_vals):
            new_dict[key] = val

        aircraft_file[self.aircraft_name] = new_dict

        # open the file as writing type
        f = open(data_path, 'w')
        json.dump(aircraft_file, f)

    # load coefficients
    def load_normal_aircraft_config(self, data_path):
        """

        :param data_path:
        :return:
        """

        f = open(data_path, 'r')
        aircraft_file = json.load(f)[self.aircraft_name]

        self.theta_front = aircraft_file['theta_front']
        self.theta_back = aircraft_file['theta_back']
        self.ts_coef = aircraft_file['ts_coef']

    # main function
    def run_airshape(self, data_path, drawing=False, tuning=False):
        """

        :param data_path:
        :param drawing:
        :param tuning:
        :return:
        """

        if tuning:

            self.tuning_surface_area(data_path)

        else:
            # load tuning coefficients
            self.load_normal_aircraft_config(data_path)
            # determine the cross area distribution
            self.compose_cross_area_dist()

        # calculate total aircraft volume
        self.calc_volume()

        # determine the wing shape (Main, Horizontal, Vertical)
        self.define_wing_config()

        # determine engine shape
        self.define_engine_config()

        # determine distributed fan shape
        if self.engine_weight_class.distributed_fan_length != 0:
            self.define_distributed_fan_config()
            if self.distfan_mount == 'upper':
                self.define_shelter()

        if drawing:
            # draw the shape on the window
            self.draw_pictures()










############################################################################################################

# Blended Wing Body shape class
class BlendedWingBodyShape(InitAirParams):

    def __init__(self, aircraft_name, init_mission_class, thermal_design_variables, engine_weight_class, init_shape_params, database_args,
                 other_args):

        engine_amplitude = other_args

        self.name = 'BWB'

        aircraft_data_path, engine_data_path, mission_data_path = database_args

        super().__init__(aircraft_name, init_mission_class, engine_amplitude)

        # baseline aircraft main wing config
        # calculate root and tip of wing chord
        self.main_croot = self.Swref / (self.BW * (1.0 + self.Lambda))
        self.main_ctip = self.Lambda * self.main_croot

        # horizontal wing (In case of bwb, no horizontal wing)
        self.hori_croot = 0
        self.hori_ctip = self.Lambda * self.hori_croot

        self.vert_croot = 0
        self.vert_ctip = self.Lambda * self.vert_croot


        # shape params [u1, v1, u2, v2]
        self.shape_params = thermal_design_variables[-5:]

        # alreday built
        self.engine_weight_class = engine_weight_class

        # unit change
        self.kg_to_lb = 2.20462
        self.m_to_ft = 3.2804

        # init shape params: [cockpit width, cockpit length(l0)]
        self.cockpit_width = 1.83  # [m]
        self.cockpit_width *= self.m_to_ft
        self.l0 = 0  # cockpit length
        aircraft_data = open(aircraft_data_path, 'r')
        aircraft_data = json.load(aircraft_data)[self.aircraft_name]
        self.l0 = aircraft_data['overall_length'] / 3 * self.m_to_ft
        # designable
        self.designable = True
        # deck level
        self.deck_level = 2  # default

        a = self.shape_params[1] * 0.5 + self.shape_params[1] * self.shape_params[2] + 0.5 * self.shape_params[4] * (self.shape_params[1] + self.shape_params[3])
        b = self.cockpit_width * 0.5
        c = self.lf * self.df / self.deck_level

        self.l0 = (-b + np.sqrt(b ** 2 + 4 * a * c)) / (2 * a)
        if np.isnan(self.l0):
            self.designable = False
        # in the future, init_shape_params include some values.
        # so if you want to study more in detail on the aircraft shape,
        # you have to modify this argument's type into list
        # For example, [coef1, coef2,...,]
        self.cabin_ratio = init_shape_params

        # cabin height
        self.cabin_height = 1.93 * self.deck_level  # [m]
        self.cabin_height *= self.m_to_ft
        self.angle_cabin = 80  # cabin angle at YZ plane
        self.cabin_length = 0  # overall cabin length

        # split the fuselage into three section (cockpit, passenger area, backend)
        self.section_lengths = []

        section1_x, section1_y = self.cockpit_width, self.l0
        self.section_lengths.append([section1_x, section1_y])
        # section2 length x,y
        section2_x, section2_y = self.shape_params[2] * self.l0, self.shape_params[1] * self.l0
        self.section_lengths.append([section2_x, section2_y])
        # section3 length x,y
        section3_x, section3_y = self.shape_params[4] * self.l0, self.shape_params[3] * self.l0
        self.section_lengths.append([section3_x, section3_y])

        # calculate overall cabin length
        self.cabin_length = section1_y + section2_y + section3_y

        # calculate total bottom area
        self.total_bottom_area = 0

        # section1
        section1_area = (self.section_lengths[0][0] + self.section_lengths[1][0]) * self.section_lengths[0][1] * 0.5
        # section2
        section2_area = self.section_lengths[1][0] * self.section_lengths[1][1]
        # section3
        section3_area = (self.section_lengths[1][0] + self.section_lengths[2][0]) * self.section_lengths[2][1] * 0.5

        # print(' Area ')
        # print('section1:', section1_area)
        # print('section2:', section2_area)
        # print('section3:', section3_area)
        # print('')

        self.total_bottom_area = section1_area + section2_area + section3_area

        # cabin shape coefficients (eclipse coefficients (a,b) , center coordinates(x_center, y_center))
        self.cabin_shape_upper_coef = None
        self.cabin_shape_lower_coef = None
        # other cabin indexes (according to passenger num)
        self.h1 = 0.1  # [m]
        self.h1 *= self.m_to_ft
        self.h3 = 0.1  # [m]
        self.h3 *= self.m_to_ft
        self.h_struct = None

        # overall wing fuselage chord
        self.overall_chord = None

        # Leading edge and Trailing edge length
        self.length_le = None
        self.length_te = None

        # Blended wing body shape coefficients (Now not use)
        # self.shape_coefs = [0.7, 1.0, 1.1]

        # Initialize cross area distribution
        self.upper_cross_area_dist = []
        self.lower_cross_area_dist = []

        # surface area of fuselage and volume of fuselage
        self.fuselage_surface_area = None
        self.volume = None
        self.main_wing_wet_area = 0  # wet area when we see at the top
        self.hori_wing_wet_area = None
        self.vert_wing_wet_area = None
        self.top_wet_area = 0
        self.side_wet_area = 0
        self.fuselage_wet_area = 0  # wet area when we see at the side
        self.wet_area = 0  # wet area

        # distributed fan mounting positions
        self.dist_fan_point = None
        self.zinit = 0

    # define fuselage airfoil type
    def define_airfoil(self, x_chord, airfoil_type=4):
        args = [x_chord, airfoil_type, self.tc, self.tcroot]
        yc, yt, rt = calc_airfoil(args)

        return yc, yt, rt

    # define passenger section(cabin)
    # ToDo: in order to have more varieties of aircraft shape, we have to add more coefficients and the method of calculating coordinates
    def define_cabin_shape(self):
        # cabin_index = 1

        # helper function for calculating eclipse coefficients
        bubl = 3 / 4
        # Upper
        bu = 0.5 * self.cabin_height + self.h1
        au = (0.5 * self.cockpit_width - self.cabin_height / np.tan(self.angle_cabin * np.pi / 180.0)) / np.sqrt(
            1.0 - (1.0 / (1.0 + 2 * self.h1 / self.cabin_height) ** 2))
        # Lower
        bl = 0.5 * self.cabin_height + self.h3
        al = (0.5 * self.cockpit_width) / np.sqrt(1.0 - (1.0 / (1.0 + 2 * self.h3 / self.cabin_height) ** 2))

        self.cabin_shape_upper_coef = au / bu
        self.cabin_shape_lower_coef = al / bl

        self.h_struct = bl + bu

        # print(self.h_struct, self.cabin_height)

        # by using structure height, determine airfoil
        # Initialize chord
        chord = 400
        chord_step = 10
        self.pitchwise = 0.3  # p / 2
        restarget = 0.0
        restargetold = 0.0
        count = 0

        while True:
            _, y_target, _ = self.define_airfoil(self.pitchwise)

            y_target *= chord

            restarget = 1.0 - y_target * 2 / self.h_struct

            # print('diff:', restarget, 'y_target:', y_target, 'h_struct:', self.h_struct)

            if count == 3000:
                break

            if abs(restarget) < 1.0e-5:
                break

            if restarget * restargetold < 0.0:
                chord_step *= 0.5

            restargetold = restarget

            chord += np.sign(restarget) * chord_step

            count += 1

        # print('chord:', chord, 'cabin length:', self.cabin_length)

        # length of LE
        self.overall_chord = chord
        self.length_le = chord * self.pitchwise  # c1
        self.length_te = chord - (self.length_le + self.cabin_length)  # c2
        self.main_croot = self.main_croot * np.sqrt((self.overall_chord / self.lf))
        self.dist_fan_point = self.length_le + self.l0 + self.section_lengths[1][0]  # section3
        print('chord:', chord, 'cockpit width:', self.l0)
        print('Leading Edge:', self.length_le, 'cabin:', self.cabin_length, 'Trailing Edge:', self.length_te)

        # confirm whether current aircraft is designable or not
        if self.length_le < 0 or self.length_te < 0:
            self.designable = False

    # helper function: calculate eclipse around length
    def calc_eclipse_around_length(self, upper_eclipse_coefs, lower_eclipse_coefs):
        au_coef, bu_coef = upper_eclipse_coefs
        al_coef, bl_coef = lower_eclipse_coefs

        # print(upper_eclipse_coefs, lower_eclipse_coefs)

        # Initialize eclipse around_length
        eclipse_ar_length = 0
        # integration coefficient
        eps_u = (bu_coef / au_coef)
        eps_l = (bl_coef / al_coef)

        if np.isnan(eps_u) or np.isnan(eps_l):
            return eclipse_ar_length

        args = [eps_u, eps_l, au_coef, al_coef, self.overall_chord]
        eclipse_ar_length = eclipse_around_length_bwb(args)

        return eclipse_ar_length

    # compose cross sectional area distribution
    def compose_cross_area_dist(self):
        # (x,y,z) 3 dimension
        # eclipse delta volume and surface area
        self.delta_eclipse_volume = []
        self.delta_eclipse_surface_area = []

        # cabin body expand coefficient
        self.cabin_expand_coef = 1.1

        cabin_outshape_datas = []

        # new code
        # design variables for bwb
        self.t1 = 10  # thickness 1
        self.t2 = 5  # thickness 2
        self.theta1 = 5
        self.theta2 = 80

        self.scoef = 0.6

        # build bezier curve class
        cp1 = [[0, 0], [0, 0.5 * self.cockpit_width + self.t1 - self.length_le * np.tan(self.theta1 * np.pi / 180.0)], [self.length_le, 0.5 * self.cockpit_width + self.t1], [self.length_le + self.l0, 0.5 * self.section_lengths[0][1]], [self.length_le + self.l0 + self.scoef * self.section_lengths[1][0], 0.5 * self.section_lengths[1][1] + self.scoef * self.section_lengths[1][0] / np.tan(self.theta * np.pi / 180.0)]]
        bc = BezierCurve(cp1)
        xycoord1 = bc.run()
        cabin_outshape_datas.extend(xycoord1)

        cp2 = [cp1[-1], [self.length_le + self.l0 + self.scoef * self.section_lengths[1][0] + self.main_croot, self.scoef * self.section_lengths[1][0] / np.tan(self.theta * np.pi / 180.0) + 0.5 * self.section_lengths[1][1]]]
        bc = BezierCurve(cp2)
        xycoord2 = bc.run()
        cabin_outshape_datas.extend(xycoord2)

        cp3 = [cp2[-1], [self.length_le + self.l0 + self.section_lengths[1][0], 0.5 * self.section_lengths[1][1]], [self.length_le + self.l0 + self.section_lengths[1][0] + self.section_lengths[2][0] + self.length_te, 0.5 * self.section_lengths[2][1] + self.t2 - self.length_te / np.tan(self.theta2 * np.pi / 180.0)], [self.length_le + self.l0 + self.section_lengths[1][0] + self.section_lengths[2][0] + self.length_te, 0]]
        bc = BezierCurve(cp3)
        xycoord3 = bc.run()
        cabin_outshape_datas.extend(xycoord3)

        if any([c < 0 for c in np.array(cabin_outshape_datas)[:, 1]]):
            self.designable = False

        # xy plane drawing
        # plt.figure()
        # cod = np.array(cabin_outshape_datas)
        # plt.plot(cod[:, 0], cod[:, 1])
        # plt.show()

        cabin_shape_datas = [[self.length_le, 0], [self.length_le, 0.5 * self.cockpit_width], [self.length_le + self.l0, self.section_lengths[1][1]], [self.length_le + self.l0 + self.section_lengths[1][0], self.section_lengths[1][1]], [self.length_le + self.l0 + self.section_lengths[1][0] + self.section_lengths[2][0], self.section_lengths[2][1]], [self.length_le + self.l0 + self.section_lengths[1][0] + self.section_lengths[2][0] + self.length_te, 0]]

        # upper and lower ratio
        bubl = 3 / 4
        delta_x = 1.0 / len(cabin_outshape_datas)

        args = [bubl, delta_x, self.tc, self.tcroot, cabin_outshape_datas, self.dist_fan_point, self.overall_chord, self.cabin_expand_coef, self.top_wet_area, self.side_wet_area, self.delta_eclipse_surface_area, self.delta_eclipse_volume, self.upper_cross_area_dist, self.lower_cross_area_dist]

        self.top_wet_area, self.side_wet_area, self.delta_eclipse_surface_area, self.delta_eclipse_volume, self.upper_cross_area_dist, self.lower_cross_area_dist, outside_shape = calc_cabin_outside_shape(args)

        # print('Upper:')
        # print(self.upper_cross_area_dist)
        # print('lower:')
        # print(self.lower_cross_area_dist)

        self.mass_inside_shape = cabin_shape_datas
        self.mass_outside_shape = outside_shape

    # define main wing configuration
    def compose_main_wing_config(self):
        # fuselage config
        fuselage_coord = np.concatenate([np.array(self.upper_cross_area_dist), np.array(self.lower_cross_area_dist)],
                                        axis=0)
        self.fus_x = fuselage_coord[:, 0]  # X coord
        self.fus_y = fuselage_coord[:, 1]  # Y coord
        self.fus_z = fuselage_coord[:, 2]  # Z coord

        # fuselage width of Blended Wing Body
        self.bwb_df = np.max(self.fus_y) * 2
        # print('fuselage bwb:', self.bwb_df)

        # main wing thickness
        self.main_troot = self.main_croot * self.tcroot
        self.main_ttip = self.main_troot * 0.7

        # bwb wing span(BW)
        self.bwb_BW = (self.mw * 0.5 - self.df * 0.5)

        # bwb wing area (exposed)
        self.bwb_Sw = self.main_croot * (self.bwb_BW * 2 + self.section_lengths[1][1] * 2 + self.t1 * 2) * (1.0 + self.Lambda) * 2

        # aspect ratio
        BW = self.bwb_BW * 2
        df = (self.section_lengths[1][1] + self.t1) * 2
        self.bwb_AR = (BW + df) ** 2 / (self.bwb_Sw)  # self.AR
        # print('bwb_AR:', self.bwb_AR, 'bwb_Swref:', self.bwb_Sw)

        self.upper_wing_area_dist = []
        self.lower_wing_area_dist = []

        # mounting position
        self.wing_mounting_position = self.length_le + self.l0 + self.section_lengths[1][0] * self.scoef

        args = [self.wing_mounting_position, self.main_ctip, self.main_croot, self.main_troot, self.main_ttip, self.bwb_BW, self.bwb_df, self.theta, self.upper_wing_area_dist, self.lower_wing_area_dist]
        self.upper_wing_area_dist, self.lower_wing_area_dist = bwb_main_wing(args)


    def define_engine_config(self):
        # the region of section3
        # self.engine_mounting_positions = self.length_le + self.l0 + self.section_lengths[1][0]

        self.upper_engine_area_dists = []
        self.lower_engine_area_dists = []

        # coef
        engine_mounting_coefx = 1.2
        engine_mounting_coefy = 2.0

        # mounting position
        self.engine_mounting_position_y = engine_mounting_coefy * (0.5 * self.bwb_BW) + 0.5 * self.bwb_df
        self.engine_mounting_position_x = engine_mounting_coefx * self.main_croot + self.wing_mounting_position

        # calculate inlet diameter and out diameter
        if self.engine_weight_class.inlet_diameter[0, 10] <= 0.0:
            front_index = 20
        else:
            front_index = 10

        args = [self.engine_weight_class, front_index, self.main_troot, self.engine_mounting_position_x, self.engine_mounting_position_y, self.upper_engine_area_dists, self.lower_engine_area_dists, self.m_to_ft]

        self.upper_engine_area_dists, self.lower_engine_area_dists = bwb_engine(args)


    def define_distributed_fan_config(self):
        self.upper_distributed_fan_dists = []
        self.lower_distributed_fan_dists = []

        df_diam_in = self.engine_weight_class.fan_in_diameter[0] * self.m_to_ft
        df_diam_out = self.engine_weight_class.fan_out_diameter[0] * self.m_to_ft

        # fan number
        Nfan = int(self.engine_weight_class.Nfan)
        distributed_engine_coefy = 0.4  #

        self.distributed_fan_mounting_positions_x = self.dist_fan_point
        self.distributed_fan_mounting_positions_y = distributed_engine_coefy * self.section_lengths[1][1] * 0.5

        args = [self.engine_weight_class, Nfan, self.distributed_fan_mounting_positions_x, self.distributed_fan_mounting_positions_y, self.main_troot, self.zinit, df_diam_in, df_diam_out, self.upper_distributed_fan_dists, self.lower_distributed_fan_dists, self.m_to_ft]

        self.distributed_fan_mounting_positions_y, self.upper_distributed_fan_dists, self.lower_distributed_fan_dists = bwb_distributed_fan(args)


    def draw_pictures(self, dimension=3, direction='xy'):

        if self.designable:

            # fuselage
            all_data_upper = np.array(self.upper_cross_area_dist)
            data_xub, data_yub, data_zub = all_data_upper[:, 0], all_data_upper[:, 1], all_data_upper[:, 2]

            all_data_lower = np.array(self.lower_cross_area_dist)
            data_xlb, data_ylb, data_zlb = all_data_lower[:, 0], all_data_lower[:, 1], all_data_lower[:, 2]

            # wing
            all_data_upper = np.array(self.upper_wing_area_dist)
            data_xu, data_yu, data_zu = all_data_upper[:, 0], all_data_upper[:, 1], all_data_upper[:, 2]

            all_data_lower = np.array(self.lower_wing_area_dist)
            data_xl, data_yl, data_zl = all_data_lower[:, 0], all_data_lower[:, 1], all_data_lower[:, 2]

            # engine
            all_data_upper = np.array(self.upper_engine_area_dists)
            data_en_xu, data_en_yu, data_en_zu = all_data_upper[:, 0], all_data_upper[:, 1], all_data_upper[:, 2]

            all_data_lower = np.array(self.lower_engine_area_dists)
            data_en_xl, data_en_yl, data_en_zl = all_data_lower[:, 0], all_data_lower[:, 1], all_data_lower[:, 2]

            # distributed fan
            if self.engine_weight_class.distributed_fan_length != 0:
                all_data_upper = np.array(self.upper_distributed_fan_dists)
                data_dist_xu, data_dist_yu, data_dist_zu = all_data_upper[:, 0], all_data_upper[:, 1], all_data_upper[:, 2]

                all_data_lower = np.array(self.lower_distributed_fan_dists)
                data_dist_xl, data_dist_yl, data_dist_zl = all_data_lower[:, 0], all_data_lower[:, 1], all_data_lower[:, 2]


            if dimension == 3:
                fig = plt.figure(figsize=(10, 8))

                ax = Axes3D(fig)

                ax.scatter(data_xu, data_yu, data_zu)
                ax.scatter(data_xl, data_yl, data_zl)
                ax.scatter(data_xub, data_yub, data_zub)
                ax.scatter(data_xlb, data_ylb, data_zlb)
                ax.scatter(data_en_xu, data_en_yu, data_en_zu)
                ax.scatter(data_en_xl, data_en_yl, data_en_zl)

                if self.engine_weight_class.distributed_fan_length != 0:
                    ax.scatter(data_dist_xu, data_dist_yu, data_dist_zu)
                    ax.scatter(data_dist_xl, data_dist_yl, data_dist_zl)

                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_xlim([-10, 130])
                ax.set_ylim([-70, 70])
                ax.set_zlim([-60, 60])
                plt.show()

            elif dimension == 2:
                if direction == 'xy':
                    plt.figure()
                    plt.scatter(data_xu, data_yu)
                    plt.scatter(data_xl, data_yl)
                    plt.scatter(data_xub, data_yub)
                    plt.scatter(data_xlb, data_ylb)

                    plt.xlabel('X')
                    plt.ylabel('Y')
                    plt.show()

                elif direction == 'yz':
                    plt.figure()
                    plt.scatter(data_yu, data_zu)
                    plt.scatter(data_yl, data_zl)
                    plt.scatter(data_yub, data_zub)
                    plt.scatter(data_ylb, data_zlb)

                    plt.xlabel('Y')
                    plt.ylabel('Z')
                    plt.ylim([-20, 20])
                    plt.show()

                elif direction == 'xz':
                    plt.figure()
                    plt.scatter(data_xu, data_zu)
                    plt.scatter(data_xl, data_zl)
                    plt.scatter(data_xub, data_zub)
                    plt.scatter(data_xlb, data_zlb)

                    plt.xlabel('X')
                    plt.ylabel('Z')
                    plt.ylim([-20, 20])
                    plt.show()

    # calculate fuselage volume and surface area
    def calc_fuselage_volume_and_surface_area(self):
        # calculate volume and surface area
        # applying numerical integration
        # ToDo we have to implement numerical integration for calculating volume and surface. Before Implementing, we have to establish the numerical equations and incrementalization on the paper.

        print('')
        print('-' * 5 + ' Blended Wing Body Results ' + '-' * 5)
        # total volume
        self.volume = np.sum(self.delta_eclipse_volume)
        print('Volume:')
        print(self.volume)
        # total surface area
        self.fuselage_surface_area = np.sum(self.delta_eclipse_surface_area)
        print('Surface area')
        print(self.fuselage_surface_area)
        self.main_wing_wet_area = 0.5 * (self.main_croot * self.lf / self.overall_chord) * self.BW * (1.0 + self.Lambda)
        print('top wet area:', self.top_wet_area)
        print('side wet area:', self.side_wet_area)
        print('total bottom area:', self.total_bottom_area)
        print('reference total bottom area:', self.lf * self.df * 0.5)
        self.wet_area = 1.7 * (self.top_wet_area + self.side_wet_area)
        self.fuselage_wet_area = self.wet_area
        print('total wet area:', self.wet_area)
        print('-' * 40)
        print('')

    # main function
    def run_airshape(self, drawing=False, draw_args=[3, 'xy']):
        # determine cabin shape
        self.define_cabin_shape()

        # determine cross area distribution
        self.compose_cross_area_dist()

        # determine main wing configuration
        self.compose_main_wing_config()

        # calculate fuselage volume and surface area
        self.calc_fuselage_volume_and_surface_area()

        # draw blended wing body shape
        self.define_engine_config()

        if self.engine_weight_class.distributed_fan_length != 0:
            self.define_distributed_fan_config()

        if drawing:
            dimension, direction = draw_args
            self.draw_pictures(dimension=dimension, direction=direction)



# test code
def test_normal():
    aircraft_name = 'A320'
    engine_name = 'V2500'
    aircraft_type = 'normal'
    propulsion_type = 'turbofan'

    # data base
    aircraft_data_path = './DataBase/aircraft_test.json'
    engine_data_path = './DataBase/engine_test.json'
    mission_data_path = './Missions/fuelburn18000.json'

    data_base_args = [aircraft_data_path, engine_data_path, mission_data_path]

    # build init mission class
    init_mission_class = InitMission(aircraft_name, engine_name, aircraft_data_path, engine_data_path)
    # set mission parameters
    # init_mission_class.load_mission_config(mission_data_path)

    # build design variable class
    dv = DesignVariable(propulsion_type, aircraft_type)
    dv_list = [4.7, 30.0, 1.61, 1380, 1.3, 0.5, 0.5, 0.5, 0.4]
    # thermal design variable class
    thermal_design_variables = dv.set_design_variable(dv_list)

    print(thermal_design_variables[-5:])

    # set off design point parameters
    off_altitude = 0
    off_mach = 0
    off_required_thrust = 133000  # [N]
    off_param_args = [off_altitude, off_mach, off_required_thrust]

    # design point parameters
    design_point_params = [10668, 0.78]

    tuning_args = [init_mission_class.required_thrust, False]
    calc_design_point_class = calc_design_point(aircraft_name, engine_name, aircraft_type, propulsion_type,
                                                thermal_design_variables, init_mission_class, design_point_params,
                                                data_base_args, tuning_args)

    rev_args = [0.98, 1.0]

    # build calc off design point class
    calc_off_design_point_class = calc_off_design_point(aircraft_name, engine_name, aircraft_type, propulsion_type,
                                                        thermal_design_variables, off_param_args, design_point_params,
                                                        data_base_args, calc_design_point_class, rev_args)

    # build engine weight class
    args = [aircraft_name, engine_name, aircraft_type, propulsion_type, engine_data_path, calc_design_point_class,
            calc_off_design_point_class]
    engine_weight_class = calc_engine_weight(args)


    # engine_amplitude
    engine_amplitude = 1.0
    other_args = engine_amplitude

    # engine position params
    engine_coef_x = 0.2
    engine_coef_y = 0.2
    engine_mounting_positions = [engine_coef_x, engine_coef_y]

    # define normal shape class
    ns = NormalShape(aircraft_name, init_mission_class, engine_weight_class, engine_mounting_positions, other_args)

    ns.run_airshape(aircraft_data_path, drawing=False, tuning=True)


    # document
    """
    # if you finished tuning, you have to use this part
    #########################################################
    # load normal aircraft shape data (if you finished tuning)
    ns.load_normal_aircraft_config(aircraft_data_path)

    # establish the distribution of surface area or length
    ns.compose_cross_area_dist()
    ########################################################

    # tuning
    # ns.tuning_surface_area(aircraft_data_path)

    # calculate total aircraft volume
    ns.calc_volume()

    # determine the wing shape (Main, Horizontal, Vertical)
    ns.define_wing_config()

    # determine engine shape
    ns.define_engine_config()

    # draw the shape on the window
    ns.draw_pictures()
    """

    print('fuselage length:', ns.fuselage_length)
    print('fuselage volume:', ns.fuselage_volume)
    print('fuselage surface area:', ns.fuselage_surface_area)
    print('fuselage wet area:', ns.fuselage_wet_area)


# test for Blended wing Body
def test_bwb():
    aircraft_name = 'A320'
    engine_name = 'V2500'
    aircraft_type = 'BWB'
    # propulsion_type = 'turbofan'
    propulsion_type = 'TeDP'

    # data base
    aircraft_data_path = './DataBase/aircraft_test.json'
    engine_data_path = './DataBase/engine_test.json'
    mission_data_path = './Missions/fuelburn18000.json'

    data_base_args = [aircraft_data_path, engine_data_path, mission_data_path]

    # build init mission class
    init_mission_class = InitMission(aircraft_name, engine_name, aircraft_data_path, engine_data_path)
    # set mission parameters
    # init_mission_class.load_mission_config(mission_data_path)

    # build design variable class
    dv = DesignVariable(propulsion_type, aircraft_type)
    # dv_list = [4.7, 30.0, 1.61, 1380, 1.4, 0.5, 0.8, 0.4, 0.1]  # turbofan + BWB

    dv_list = [40.0, 1430, 5.0, 1.24, 0.15, 0.99, 3, 1.0, 0.8, 0.48, 1.0, 0.2]
    # design variables for blended wing body are determined by displacement of cabin

    # thermal design variable class
    thermal_design_variables = dv.set_design_variable(dv_list)
    print(thermal_design_variables[-5:])

    # set off design point parameters
    off_altitude = 0
    off_mach = 0
    off_required_thrust = 133000  # [N]
    off_param_args = [off_altitude, off_mach, off_required_thrust]

    # design point parameters
    design_point_params = [10668, 0.78]

    tuning_args = [init_mission_class.required_thrust, False]
    calc_design_point_class = calc_design_point(aircraft_name, engine_name, aircraft_type, propulsion_type,
                                                thermal_design_variables, init_mission_class, design_point_params,
                                                data_base_args, tuning_args)

    rev_args = [1.391, 1.0]

    # build calc off design point class
    calc_off_design_point_class = calc_off_design_point(aircraft_name, engine_name, aircraft_type, propulsion_type,
                                                        thermal_design_variables, off_param_args, design_point_params,
                                                        data_base_args, calc_design_point_class, rev_args)

    # build engine weight class
    args = [aircraft_name, engine_name, aircraft_type, propulsion_type, engine_data_path, calc_design_point_class,
            calc_off_design_point_class]
    engine_weight_class = calc_engine_weight(args)


    # engine_amplitude
    engine_amplitude = 1.0
    other_args = engine_amplitude

    # init shape params
    cabin_ratio = 0.8  # h1 / h3
    init_shape_params = cabin_ratio

    # define normal shape class
    bwbs = BlendedWingBodyShape(aircraft_name, init_mission_class, thermal_design_variables, engine_weight_class, init_shape_params,
                                data_base_args, other_args)

    # draw
    drawing = False
    draw_args = [3, 'xy']

    # run main function
    bwbs.run_airshape(drawing, draw_args)


def test_normal_distfan():
    aircraft_name = 'A320'
    engine_name = 'V2500'
    aircraft_type = 'normal'
    propulsion_type = 'TeDP'

    # data base
    aircraft_data_path = './DataBase/aircraft_test.json'
    engine_data_path = './DataBase/engine_test.json'
    mission_data_path = './Missions/fuelburn18000.json'

    data_base_args = [aircraft_data_path, engine_data_path, mission_data_path]

    # build init mission class
    init_mission_class = InitMission(aircraft_name, engine_name, aircraft_data_path, engine_data_path)
    # set mission parameters
    init_mission_class.load_mission_config(mission_data_path)

    # build design variable class
    dv = DesignVariable(propulsion_type, aircraft_type)
    dv_list = [40.0, 1430, 5.0, 1.24, 0.7, 0.99, 3, 1.3, 0.5, 0.5, 0.5, 0.4]
    # thermal design variable class
    thermal_design_variables = dv.set_design_variable(dv_list)

    print(thermal_design_variables[-5:])

    # set off design point parameters
    off_altitude = 0
    off_mach = 0
    off_required_thrust = 133000  # [N]
    off_param_args = [off_altitude, off_mach, off_required_thrust]
    # design point parameters
    design_point_params = [10668, 0.78]

    # build calc design point class
    tuning_args = [init_mission_class.required_thrust, False]
    calc_design_point_class = calc_design_point(aircraft_name, engine_name, aircraft_type, propulsion_type,
                                              thermal_design_variables, init_mission_class, design_point_params,
                                              data_base_args, tuning_args)

    rev_args = [1.391, 1.0]

    # build calc off design point class
    calc_off_design_point_class = calc_off_design_point(aircraft_name, engine_name, aircraft_type, propulsion_type, thermal_design_variables, off_param_args, design_point_params, data_base_args, calc_design_point_class, rev_args)

    # build engine weight class
    args = [aircraft_name, engine_name, aircraft_type, propulsion_type, engine_data_path, calc_design_point_class, calc_off_design_point_class]
    engine_weight_class = calc_engine_weight(args)

    # engine_amplitude
    engine_amplitude = 1.0
    other_args = engine_amplitude

    # engine position params
    engine_coef_x = 0.2
    engine_coef_y = 0.2
    engine_mounting_positions = [engine_coef_x, engine_coef_y]

    # define normal shape class
    ns = NormalShape(aircraft_name, init_mission_class, engine_weight_class, engine_mounting_positions, other_args)

    ns.run_airshape(aircraft_data_path, drawing=False, tuning=False)


    # document
    """
    # if you finished tuning, you have to use this part
    #########################################################
    # load normal aircraft shape data (if you finished tuning)
    ns.load_normal_aircraft_config(aircraft_data_path)

    # establish the distribution of surface area or length
    ns.compose_cross_area_dist()
    ########################################################

    # tuning
    # ns.tuning_surface_area(aircraft_data_path)

    # calculate total aircraft volume
    ns.calc_volume()

    # determine the wing shape (Main, Horizontal, Vertical)
    ns.define_wing_config()

    # determine engine shape
    ns.define_engine_config()

    # draw the shape on the window
    ns.draw_pictures()
    """

    print('fuselage length:', ns.fuselage_length)
    print('fuselage volume:', ns.fuselage_volume)
    print('fuselage surface area:', ns.fuselage_surface_area)
    print('fuselage wet area:', ns.fuselage_wet_area)


if __name__ == '__main__':
    start = time.time()
    # test_normal()
    test_bwb()
    # test_normal_distfan()
    finish = time.time()
    print(finish - start)
