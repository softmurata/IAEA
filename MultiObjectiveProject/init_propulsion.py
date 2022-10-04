from EngineComponent import *
from init_condition import InitPhysicCondition
from init_engine import SetElementEff
import numpy as np
from standard_air_utils import StandardAir


# class for engine efficient


# class for defining Required components
class InitPropulsion(object):

    def __init__(self):

        ######################################################################################################################################################################################
        # initialize components they are restoring class object
        # if you find new types of propulsion system,you should add components of it to this position
        # [[Core],[Elec]]
        self.turbojet_components = [[Inlet, LPC, HPC, CC, HPT, HPTCool, LPT, LPTCool, CoreOut, Nozzle, Jet], []]

        self.turboshaft_components = [[Inlet, LPC, HPC, CC, HPT, HPTCool, LPT, LPTCool, CoreOut, Nozzle, Jet], []]

        self.turbofan_components = [
            [Inlet, Fan, LPC, HPC, CC, HPT, HPTCool, LPT, LPTCool, CoreOut, Nozzle, Jet, FanNozzle, FanJet], []]

        self.TeDP_components = [[Inlet, LPC, HPC, CC, HPT, HPTCool, LPT, LPTCool, CoreOut, Nozzle, Jet],
                                [InletElec, FanElec, FanNozzleElec, FanJetElec]]

        self.PartialElectric_components = [
            [Inlet, Fan, LPC, HPC, CC, HPT, HPTCool, LPT, LPTCool, CoreOut, Nozzle, Jet, FanNozzle, FanJet],
            [InletElec, FanElec, FanNozzleElec, FanJetElec]]

        self.battery_components = [[], [InletElec, FanElec, FanNozzleElec, FanJetElec]]

        self.hybridtubojet_components = [[Inlet, LPC, HPC, CC, HPT, HPTCool, LPT, LPTCool, CoreOut, Nozzle, Jet],
                                         [InletElec, FanElec, FanNozzleElec, FanJetElec]]

        self.hybridturbofan_components = [
            [Inlet, Fan, LPC, HPC, CC, HPT, HPTCool, LPT, LPTCool, CoreOut, Nozzle, Jet, FanNozzle, FanJet],
            [InletElec, FanElec, FanNozzleElec, FanJetElec]]

    #######################################################################################################################################################################################

    def get_component_classes(self, propulsion_type):

        # Initialize
        selected_components = None

        if propulsion_type == 'turbojet':

            selected_components = self.turbojet_components

        elif propulsion_type == 'turboshaft':

            selected_components = self.turboshaft_components

        elif propulsion_type == 'turbofan':

            selected_components = self.turbofan_components

        elif propulsion_type == 'TeDP':

            selected_components = self.TeDP_components

        elif propulsion_type == 'PartialElectric':

            selected_components = self.PartialElectric_components

        elif propulsion_type == 'battery':

            selected_components = self.battery_components

        elif propulsion_type == 'hybridturbojet':

            selected_components = self.hybridtubojet_components

        elif propulsion_type == 'hybridturbofan':

            selected_components = self.hybridturbofan_components

        return selected_components


# 定義 設計変数の違いにより格納する部分を変化させる必要がある
###############################################################################################################################################
# dref 0:altitude 1:mach_number(cruise) 4:static temp 5:static pressure 10:required thrust@design point 20:BPR(core) 21:OPR(core) 22:FPR(core)
#     23:distributed ratio LP by HP 24:TIT(core) 
#     30:nfan 31:BPRe 32:FPRe 33:div_alpha(分配割合) 34:nele 35:electric ratio(electric energy against energy generated by turboengine)
#
#
#
# design variables:
#                TurboJet:[OPR,TIT]
#                TurboShaft:[OPR,TIT]
#                TurboFan:[BPR,OPR,FPR,TIT]
#                TeDP:[OPR,TIT,div_alpha,nele,BPRe,FPRe,Nfan]
#                PartialElectric:[BPR,OPR,FPR,TIT,div_alpha,nele,BPRe,FPRe,Nfan]
###############################################################################################################################################


# class for defining coefficients of thermal class
class InitThermalParams(InitPhysicCondition):

    def __init__(self, init_mission_class, design_point_params, mission_data_path):

        # design point parameters
        dp_altitude, dp_mach = design_point_params

        super().__init__(dp_altitude, dp_mach)

        # all module which include in Engine Component.py
        self.modules = {
            'Inlet': 0, 'Fan': 10, 'FanNozzle': 18, 'FanJet': 19, 'LPC': 20, 'HPC': 25, 'CC': 30, 'HPT': 40,
            'HPTCool': 41, 'LPT': 45, 'LPTCool': 46, 'CoreOut': 50, 'Nozzle': 80, 'Jet': 90
        }

        # gravity
        self.g = 9.81

        # define init mission class
        self.init_mission_class = init_mission_class
        # load mission data from mission file
        self.init_mission_class.load_mission_config(mission_data_path)

        # efficient coef class
        self.see = SetElementEff()
        # list which restores coefficients constant
        self.cref = np.zeros((20, 100))  # Core
        self.cref_e = np.zeros((20, 100))  # Distributed

        # list which restores the thermal parameters
        self.dref = np.zeros(100)

        # Define design point
        self.define_design_point()

        #####Design off params######
        # Initialize core
        self.coff = self.cref
        self.doff = self.dref
        # Initialize electric
        self.coff_e = self.cref_e

    def build(self, thermal_design_variables):

        """
		:param: thermal design variables
		ex) ['BPR','OPR','FPR','TIT','BPRe','FPRe','div_alpha','nele','Nfan','electric_ratio',u1','v1','u2','v2']
		"""

        # define design variables
        self.dref[20] = thermal_design_variables[0]  # BPR(core)
        self.dref[21] = thermal_design_variables[1]  # OPR(core)
        self.dref[22] = thermal_design_variables[2]  # FPR(core)
        self.dref[24] = thermal_design_variables[3]  # TIT(core)
        self.dref[30] = thermal_design_variables[8]  # Nfan
        self.dref[31] = thermal_design_variables[4]  # BPRe(Distributed)
        self.dref[32] = thermal_design_variables[5]  # FPRe(Distributed)
        self.dref[33] = thermal_design_variables[6]  # div_alpha
        self.dref[34] = thermal_design_variables[7]  # nele
        self.dref[35] = thermal_design_variables[8]  # electric ratio

    def define_design_point(self):

        ########################################################################################
        # define dref
        self.dref[0] = self.init_mission_class.altitude  # altitude
        self.dref[1] = self.init_mission_class.mach  # Cruise Mach
        self.dref[4] = self.static_T  # static temperature
        self.dref[5] = self.static_P  # static pressure
        # aircraft mass weight ratio's product
        mass_product = np.prod(self.init_mission_class.mass_ratio[:2])
        self.dref[10] = (self.init_mission_class.max_takeoff_weight * mass_product - self.init_mission_class.fuelburn_coef * self.init_mission_class.fuel_weight) * self.g / self.init_mission_class.Lift_by_Drag / self.init_mission_class.engine_num  # required thrust at design point
        self.dref[23] = self.init_mission_class.lp_hp_dist_ratio  # LPC by HPC pressure ratio

        #######################################################################################

        # 定数の定義(Cp,gamma)
        module_number = list(self.modules.values())
        for m_num in module_number:
            # Before Combustion Chamber
            if m_num <= 25:
                # Before Compressor
                if m_num <= 19:
                    self.cref[1, m_num] = self.cp_comp_before
                    self.cref[2, m_num] = self.gamma_comp_before
                    # add electric data too
                    self.cref_e[1, m_num] = self.cp_comp_before
                    self.cref_e[2, m_num] = self.gamma_comp_before
                else:
                    self.cref[1, m_num] = self.cp_comp_before
                    self.cref[2, m_num] = self.gamma_comp_before

            elif 30 < m_num <= 90:
                self.cref[1, m_num] = self.cp_comp_after
                self.cref[2, m_num] = self.gamma_comp_after

            elif m_num == 30:
                self.cref[1, m_num] = 0.5 * (self.cp_comp_before + self.cp_comp_after)
                self.cref[2, m_num] = 0.5 * (self.gamma_comp_before + self.gamma_comp_after)

        # 熱効率
        self.eff_comp = min(0.76 + 0.04 * self.init_mission_class.tech_lev, 1.0)
        self.eff_turb = min(0.76 + 0.04 * self.init_mission_class.tech_lev, 1.0)

        self.cref[3, 0] = 1.00  # インテーク効率
        self.cref[3, 10] = self.eff_comp
        self.cref[3, 20] = self.eff_comp
        self.cref[3, 25] = self.eff_comp
        self.cref[3, 30] = self.see.eps_b  # 燃焼効率(CC)
        self.cref[3, 40] = self.eff_turb
        self.cref[3, 41] = 0.0  # mixer
        self.cref[3, 45] = self.eff_turb
        self.cref[3, 46] = 0.0  # mixer
        self.cref[3, 70] = self.see.eps_afb  # 全圧損失
        self.cref[3, 80] = self.see.pai80  # ノズル圧力比(core)
        self.cref[3, 15] = self.cref[3, 30]
        self.cref[3, 18] = self.see.pai18  # ノズル圧力比(Bypass)

        # 低位発熱量
        self.cref[4, 30] = self.init_mission_class.lhv
        self.cref[4, 40] = self.see.cool_air_hp  # HPT冷却空気割合
        self.cref[4, 45] = self.see.cool_air_lp  # LPT冷却空気割合

        # After Burner
        self.cref[4, 70] = self.init_mission_class.lhv
        self.cref[4, 80] = self.init_mission_class.lhv

        # 機械効率
        self.cref[5, 30] = self.see.yta_b
        self.cref[5, 40] = self.see.mec_hp  # HPT機械効率
        self.cref[5, 45] = self.see.mec_lp  # LPT機械効率
        self.cref[5, 70] = self.see.yta_afb  # AfterBurner効率
        self.cref[5, 15] = self.cref[5, 30]

        #####Electric part########

        self.cref_e[3, 0] = 1.00
        self.cref_e[3, 10] = self.eff_comp
        self.cref_e[3, 15] = self.cref[3, 30]
        self.cref_e[3, 18] = self.see.pai18

        self.cref[4, 20] = 1.0  # YT.electricLP
        self.cref[5, 20] = 0.0  # YT.drive
        self.cref[4, 25] = 0.0  # YT.electricHP
        self.cref[5, 25] = -0.003  # YT.drive

    def define_design_off_point(self, off_param_args):

        # off_params
        # [0.0,0.0,1.0,0.0,-0.003,1.0,1.0,1.0,1.0,1.0]
        self.set_design_off_point(off_param_args)
        altitude = self.off_params[0]
        self.doff[0] = altitude  # altitude
        self.doff[1] = self.off_params[1]  # mach number
        # Standard air class
        sa = StandardAir(altitude)
        # design off point temperature,pressure
        static_t, static_p = sa.T, sa.P

        # others
        self.doff[4] = static_t  # static temperature
        self.doff[5] = static_p  # static pressure
        self.doff[10] = self.off_params[-1]  # required thrust at design off point
        self.doff[50] = self.off_params[3]
        self.doff[51] = self.off_params[4]
        self.doff[40] = self.off_params[5]
        self.doff[41] = self.off_params[6]
        self.doff[45] = self.off_params[7]
        self.doff[42] = self.off_params[8]
        self.doff[43] = self.off_params[9]

        # Polar curv fitting coefficients
        # x**a+y**b=2 (a,b)
        ##fan##
        self.coff[4, 10] = 5.0
        self.coff[5, 10] = 10.0  # 10.0

        ##lpc##
        self.coff[4, 20] = 5.0
        self.coff[5, 20] = 10.0  # 10.0

        ##hpc##
        self.coff[4, 25] = 5.0
        self.coff[5, 25] = 10.0  # 10.0

        ##distributed fan##

        # fan
        self.coff_e[4, 10] = 5.0
        self.coff_e[5, 10] = 5.0

        # lpc
        self.coff[6, 20] = 5.0
        self.coff[7, 20] = 5.0

        # hpc
        self.coff[6, 25] = 5.0
        self.coff[7, 25] = 5.0