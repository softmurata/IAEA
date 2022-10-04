# Core engine params
class InitEngineParamsCore(object):

    def __init__(self):
        # 軸流マッハ数(Inlet)
        self.fan_inlet_mach_number = 0.6
        self.lpc_inlet_mach_number = 0.5
        self.hpc_inlet_mach_number = 0.5
        self.cc_inlet_mach_number = 0.3
        self.hpt_inlet_mach_number = 0.1
        self.hptcool_inlet_mach_number = 0.45
        self.lpt_inlet_mach_number = 0.45
        self.lptcool_inlet_mach_number = 0.45
        self.nozzle_inlet_mach_number = 0.3

        # Outlet
        self.lpc_outlet_mach_number = 0.5
        self.hpc_outlet_mach_number = 0.5
        self.hpt_outlet_mach_number = 0.45
        self.lpt_outlet_mach_number = 0.45

        # Tip周速
        self.fan_tip_velocity = (350.0, 120.0)
        self.lpc_tip_velocity = 400.0
        self.hpc_tip_velocity = 400.0
        self.hpt_tip_velocity = 480.0
        self.lpt_tip_velocity = 480.0

        # 燃焼交差面積
        self.cc_cross_mach = 0.02
        self.LB_H = 2.7

        # Tip hub 比
        self.fan_tip_hub_ratio = 0.32
        self.lpc_tip_hub_ratio = 0.5
        self.hpc_tip_hub_ratio = 0.5


# engine effient coefficeint
class SetElementEff(object):

    def __init__(self, cool_air_hp=0.15, cool_air_lp=0.0):
        # 燃焼器圧力損失
        self.eps_b = 0.06
        # 燃焼効率
        self.yta_b = 0.96
        # HPT冷却空気割合
        self.cool_air_hp = cool_air_hp
        # LPT冷却空気割合
        self.cool_air_lp = cool_air_lp

        # HPT機械効率
        self.mec_hp = 0.99
        # LPT機械効率
        self.mec_lp = 0.99

        # アフターバーナー圧力損失
        self.eps_afb = 0.0
        # アフターバーナー燃焼効率
        self.yta_afb = 1.0

        # coreノズル圧力損失(module番号80)
        self.pai80 = 0.98

        # bypassノズル圧力損失(module番号18)
        self.pai18 = 0.98
