import numpy as np
from init_engine import InitEngineParamsCore
from EngineComponent_utils import *


# from SetDesignPoint import DesignParams,SetEngineParamsCore

################Componentごとに性能計算ができるように関数を定義############################

# 係数計算
def calc_coef(cp, gamma):
    gamma_t = 0.5 * (gamma - 1.0)
    gamma_p = gamma / (gamma - 1)
    gamma_w = 0.5 * (gamma + 1.0) / (gamma - 1.0)
    rg = cp / gamma_p

    return (gamma_t, gamma_p, gamma_w, rg)


# 全温・全圧・面積を求める関数
def calc_thermal_indexes(W, TT, PT, tau, pai, mc, gamma, gamma_t, gamma_w, rg):
    TT = TT * tau
    PT = PT * pai
    MFP = np.sqrt(gamma / rg) * mc / (1.0 + gamma_t * mc ** 2) ** gamma_w
    A = W / (PT / np.sqrt(TT) * max(MFP, 1.0e-10))

    return TT, PT, MFP, A


#####################################Module Class######################################

class Inlet(object):

    def __init__(self, cref, dref, propulsion_type):
        # パラメータが入っている行列
        self.cref = cref
        self.dref = dref
        # モジュール番号
        self.module_index = 0

    def __call__(self, qref, qref_e):
        args = [qref, qref_e, self.dref, self.cref, self.module_index]
        qref = run_inlet_core(args)
        """
        gamma, cp = self.cref[2, self.module_index], self.cref[1, self.module_index]

        gamma_t, gamma_p, gamma_w, rg = calc_coef(cp, gamma)

        # 流入マッハ数
        mc00 = self.dref[1]
        T00, P00 = self.dref[4], self.dref[5]
        BPR = self.dref[20]
        # 全温・全圧
        TT00 = T00 * (1.0 + gamma_t * mc00 ** 2)
        PT00 = P00 * (1.0 + gamma_t * mc00 ** 2) ** gamma_p
        # コア側流量　1として計算する
        W00 = 1.0 + BPR
        MFP00 = np.sqrt(gamma / rg) * mc00 / (1.0 + gamma_t * mc00 ** 2) ** gamma_w
        # 面積
        A00 = W00 / (PT00 / np.sqrt(TT00) * max(MFP00, 1.0e-10))
        # 速度
        U00 = mc00 * np.sqrt(gamma * rg * TT00 / (1.0 + gamma_t * mc00 ** 2))

        # 圧力比・温度比を計算
        ytd00 = self.cref[3, self.module_index]
        pai00 = 1.0 / ((1.0 + (1.0 - ytd00) * gamma_t * mc00 ** 2) ** gamma_p)
        tau00 = 1.0

        qref[1, self.module_index] = W00
        qref[2, self.module_index] = A00
        qref[3, self.module_index] = 0.0  # 回転数
        qref[4, self.module_index] = TT00
        qref[5, self.module_index] = PT00
        qref[6, self.module_index] = 0.0  # エントロピー
        qref[7, self.module_index] = tau00
        qref[8, self.module_index] = pai00
        qref[9, self.module_index] = MFP00

        # 推力f00
        qref[0, self.module_index] = -W00 * U00
        """

        return qref


class Fan(InitEngineParamsCore):

    def __init__(self, cref, dref, propulsion_type):
        # 継承
        super().__init__()
        # パラメータが入っている行列
        self.cref = cref
        self.dref = dref

        self.propulsion_type = propulsion_type
        # モジュール番号
        self.module_index = 10

    def __call__(self, qref, qref_e):
        args = [qref, qref_e, self.dref, self.cref, self.module_index, self.fan_inlet_mach_number]
        qref = run_fan_core(args)

        """
        gamma, cp = self.cref[2, self.module_index], self.cref[1, self.module_index]

        gamma_t, gamma_p, gamma_w, rg = calc_coef(cp, gamma)

        # 流入マッハ数
        mc10 = self.fan_inlet_mach_number
        TT00, PT00 = qref[4, 0], qref[5, 0]
        tau00, pai00 = qref[7, 0], qref[8, 0]
        W00 = qref[1, 0]

        # 流量
        W10 = W00

        TT10, PT10, MFP10, A10 = calc_thermal_indexes(W10, TT00, PT00, tau00, pai00, mc10, gamma, gamma_t, gamma_w, rg)

        # 圧力比・温度比を計算
        FPR = self.dref[22]
        ytp10 = self.cref[3, self.module_index]
        pai10 = FPR
        tau10 = pai10 ** (1.0 / (gamma_p * ytp10))

        # エントロピー計算
        L10 = W10 * cp * TT10 * (tau10 - 1.0)
        ytc10 = (pai10 ** (1.0 / gamma_p) - 1.0) / (tau10 - 1.0)

        qref[1, self.module_index] = W10
        qref[2, self.module_index] = A10
        qref[3, self.module_index] = 1.0  # 回転数
        qref[4, self.module_index] = TT10
        qref[5, self.module_index] = PT10
        qref[6, self.module_index] = L10  # エントロピー
        qref[7, self.module_index] = tau10
        qref[8, self.module_index] = pai10
        qref[9, self.module_index] = MFP10

        qref[0, self.module_index] = ytc10
        """

        return qref


class LPC(InitEngineParamsCore):

    def __init__(self, cref, dref, propulsion_type):
        # 継承
        super().__init__()
        # パラメータが入っている行列
        self.cref = cref
        self.dref = dref
        # 推進形態(propulsion type)によってつながっているモジュールが異なる
        self.module_pre_index = 10  # (default)

        if propulsion_type in ['turboFan', 'PartialElectric']:

            self.module_pre_index = 10

        elif propulsion_type in ['turbojet', 'turboshaft', 'TeDP']:

            self.module_pre_index = 0

        # モジュール番号
        self.module_index = 20

        self.propulsion_type = propulsion_type

    def __call__(self, qref, qref_e):

        args = [qref, qref_e, self.dref, self.cref, self.module_index, self.module_pre_index, self.lpc_inlet_mach_number, self.lpc_outlet_mach_number, self.propulsion_type]
        qref = run_lpc_core(args)

        """
        gamma, cp = self.cref[2, self.module_index], self.cref[1, self.module_index]

        gamma_t, gamma_p, gamma_w, rg = calc_coef(cp, gamma)

        # 流入マッハ数
        mc20 = self.lpc_inlet_mach_number
        TT, PT = qref[4, self.module_pre_index], qref[5, self.module_pre_index]
        tau, pai = qref[7, self.module_pre_index], qref[8, self.module_pre_index]
        W = 1.0  # Core only

        # 流量
        W20 = W
        TT20, PT20, MFP20, A20 = calc_thermal_indexes(W20, TT, PT, tau, pai, mc20, gamma, gamma_t, gamma_w, rg)

        # 圧力比・温度比を計算
        OPR = self.dref[21]
        FPR = self.dref[22]
        CLH = self.dref[23]
        ytp20 = self.cref[3, self.module_index]

        if self.propulsion_type in ['turbofan', 'PartialElectric']:
            pai20 = (OPR / FPR) ** CLH
        elif self.propulsion_type in ['turbojet', 'turboshaft', 'TeDP']:
            pai20 = (OPR) ** CLH
        tau20 = pai20 ** (1.0 / (gamma_p * ytp20))

        # エントロピー計算
        L20 = W20 * cp * TT20 * (tau20 - 1.0)
        ytc20 = (pai20 ** (1.0 / gamma_p) - 1.0) / (tau20 - 1.0)

        qref[1, self.module_index] = W20
        qref[2, self.module_index] = A20
        qref[3, self.module_index] = 1.0  # 回転数
        qref[4, self.module_index] = TT20
        qref[5, self.module_index] = PT20
        qref[6, self.module_index] = L20  # エントロピー
        qref[7, self.module_index] = tau20
        qref[8, self.module_index] = pai20
        qref[9, self.module_index] = MFP20

        qref[0, self.module_index] = ytc20

        # 出口計算
        mc20e = self.lpc_outlet_mach_number
        TT20e = TT20 * tau20
        PT20e = PT20 * pai20
        MFP20e = np.sqrt(gamma / rg) * mc20e / (1.0 + gamma_t * mc20e ** 2) ** gamma_w
        A20e = W20 / (PT20e / np.sqrt(TT20e) * MFP20e)

        qref[11, self.module_index] = A20e
        """

        return qref


class HPC(InitEngineParamsCore):

    def __init__(self, cref, dref, propulsion_type):
        # 継承
        super().__init__()
        # パラメータが入っている行列
        self.cref = cref
        self.dref = dref
        # 推進形態(propulsion type)によってつながっているモジュールが異なる
        self.module_pre_index = 20  # (default)
        # モジュール番号
        self.module_index = 25

        self.propulsion_type = propulsion_type

    def __call__(self, qref, qref_e):
        args = [qref, qref_e, self.dref, self.cref, self.module_index, self.module_pre_index, self.hpc_inlet_mach_number, self.hpc_outlet_mach_number, self.propulsion_type]
        qref = run_hpc_core(args)

        """
        gamma, cp = self.cref[2, self.module_index], self.cref[1, self.module_index]

        gamma_t, gamma_p, gamma_w, rg = calc_coef(cp, gamma)

        # 流入マッハ数
        mc25 = self.hpc_inlet_mach_number
        TT, PT = qref[4, self.module_pre_index], qref[5, self.module_pre_index]
        tau, pai = qref[7, self.module_pre_index], qref[8, self.module_pre_index]
        W = qref[1, self.module_pre_index]

        # 流量
        W25 = W
        TT25, PT25, MFP25, A25 = calc_thermal_indexes(W25, TT, PT, tau, pai, mc25, gamma, gamma_t, gamma_w, rg)

        # 圧力比・温度比を計算
        OPR = self.dref[21]
        FPR = self.dref[22]
        ytp25 = self.cref[3, self.module_index]

        if self.propulsion_type in ['turbofan', 'PartialElectric']:

            pai25 = OPR / (FPR * pai)

        elif self.propulsion_type in ['turbojet', 'turboshaft', 'TeDP']:

            pai25 = OPR / pai

        tau25 = pai25 ** (1.0 / (gamma_p * ytp25))

        # エントロピー計算
        L25 = W25 * cp * TT25 * (tau25 - 1.0)
        ytc25 = (pai25 ** (1.0 / gamma_p) - 1.0) / (tau25 - 1.0)

        qref[1, self.module_index] = W25
        qref[2, self.module_index] = A25
        qref[3, self.module_index] = 1.0  # 回転数
        qref[4, self.module_index] = TT25
        qref[5, self.module_index] = PT25
        qref[6, self.module_index] = L25  # エントロピー
        qref[7, self.module_index] = tau25
        qref[8, self.module_index] = pai25
        qref[9, self.module_index] = MFP25

        qref[0, self.module_index] = ytc25

        # 出口計算
        mc25e = self.hpc_outlet_mach_number
        TT25e = TT25 * tau25
        PT25e = PT25 * pai25
        MFP25e = np.sqrt(gamma / rg) * mc25e / (1.0 + gamma_t * mc25e ** 2) ** gamma_w
        A25e = W25 / (PT25e / np.sqrt(TT25e) * MFP25e)

        qref[11, self.module_index] = A25e
        """

        return qref


class CC(InitEngineParamsCore):

    def __init__(self, cref, dref, propulsion_type):
        # 継承
        super().__init__()
        # パラメータが入っている行列
        self.cref = cref
        self.dref = dref
        # 推進形態(propulsion type)によってつながっているモジュールが異なる
        self.module_pre_index = 25  # (default)
        # モジュール番号
        self.module_index = 30

    def __call__(self, qref, qref_e):
        args = [qref, qref_e, self.dref, self.cref, self.module_index, self.module_pre_index, self.cc_inlet_mach_number, self.cc_cross_mach]
        qref = run_cc_core(args)

        """
        gamma, cp = self.cref[2, self.module_index], self.cref[1, self.module_index]

        gamma_t, gamma_p, gamma_w, rg = calc_coef(cp, gamma)

        # 流入マッハ数
        mc30 = self.cc_inlet_mach_number
        TT, PT = qref[4, self.module_pre_index], qref[5, self.module_pre_index]
        tau, pai = qref[7, self.module_pre_index], qref[8, self.module_pre_index]
        W = qref[1, self.module_pre_index]
        W20 = qref[1, 20]

        # 冷却空気割合
        cbr40 = self.cref[4, 40]  # LPT冷却割合
        cbr45 = self.cref[4, 45]  # HPT冷却割合

        # 流量
        W30 = W - (cbr40 + cbr45) * W20
        TT30, PT30, MFP30, A30 = calc_thermal_indexes(W30, TT, PT, tau, pai, mc30, gamma, gamma_t, gamma_w, rg)

        # 圧力比・温度比を計算
        TIT = self.dref[24]
        eps30 = self.cref[3, self.module_index]  # 燃焼圧損
        hlv30 = self.cref[4, self.module_index]  # 低位発熱量
        ytb30 = self.cref[5, self.module_index]  # 燃焼効率
        pai30 = 1.0 - eps30
        tau30 = TIT / TT30

        # 燃焼消費量の計算
        WF30 = W30 * cp * TT30 * (tau30 - 1.0) / (ytb30 * hlv30 - cp * TT30 * tau30)
        L30 = ytb30 * WF30 * hlv30

        qref[1, self.module_index] = W30
        qref[2, self.module_index] = A30
        qref[3, self.module_index] = 0.0  # 回転数
        qref[4, self.module_index] = TT30
        qref[5, self.module_index] = PT30
        qref[6, self.module_index] = L30  # エントロピー
        qref[7, self.module_index] = tau30
        qref[8, self.module_index] = pai30
        qref[9, self.module_index] = MFP30

        qref[0, self.module_index] = WF30  # 燃料消費量

        # 出口計算
        mc30e = self.cc_cross_mach
        TT30e, PT30e, MFP30e, A30e = calc_thermal_indexes(W30, TT30, PT30, tau30, pai30, mc30e, gamma, gamma_t, gamma_w,
                                                          rg)

        qref[11, self.module_index] = A30e
        """


        return qref


class HPT(InitEngineParamsCore):

    def __init__(self, cref, dref, propulsion_type):
        # 継承
        super().__init__()
        # パラメータが入っている行列
        self.cref = cref
        self.dref = dref
        # 推進形態(propulsion type)によってつながっているモジュールが異なる
        self.module_pre_index = 30  # (default)
        # モジュール番号
        self.module_index = 40

    def __call__(self, qref, qref_e):

        args = [qref, qref_e, self.dref, self.cref, self.module_index, self.module_pre_index, self.hpt_inlet_mach_number]
        qref = run_hpt_core(args)

        """
        gamma, cp = self.cref[2, self.module_index], self.cref[1, self.module_index]

        gamma_t, gamma_p, gamma_w, rg = calc_coef(cp, gamma)

        # 流入マッハ数
        mc40 = self.hpt_inlet_mach_number
        TT, PT = qref[4, self.module_pre_index], qref[5, self.module_pre_index]
        tau, pai = qref[7, self.module_pre_index], qref[8, self.module_pre_index]
        W = qref[1, self.module_pre_index]
        WF30 = qref[0, 30]

        # 流量
        W40 = W + WF30
        TT40, PT40, MFP40, A40 = calc_thermal_indexes(W40, TT, PT, tau, pai, mc40, gamma, gamma_t, gamma_w, rg)

        ytm40 = self.cref[5, self.module_index]
        ytd25 = self.cref[5, 25]
        L25 = qref[6, 25]
        # エネルギーマッチング
        L40 = -L25 / (ytm40 + ytd25)
        ytp40 = self.cref[3, self.module_index]

        # 圧力比・温度比を計算
        tau40 = 1.0 + L40 / (W40 * cp * TT40)
        pai40 = tau40 ** (gamma_p / ytp40)

        ytt40 = (tau40 - 1.0) / (pai40 ** (1.0 / gamma_p) - 1.0)

        qref[1, self.module_index] = W40
        qref[2, self.module_index] = A40
        qref[3, self.module_index] = 1.0  # 回転数
        qref[4, self.module_index] = TT40
        qref[5, self.module_index] = PT40
        qref[6, self.module_index] = L40  # エントロピー
        qref[7, self.module_index] = tau40
        qref[8, self.module_index] = pai40
        qref[9, self.module_index] = MFP40

        qref[0, self.module_index] = ytt40
        """

        return qref


class HPTCool(InitEngineParamsCore):

    def __init__(self, cref, dref, propulsion_type):
        # 継承
        super().__init__()
        # パラメータが入っている行列
        self.cref = cref
        self.dref = dref
        # 推進形態(propulsion type)によってつながっているモジュールが異なる
        self.module_pre_index = 40  # (default)
        # モジュール番号
        self.module_index = 41

    def __call__(self, qref, qref_e):

        args = [qref, qref_e, self.dref, self.cref, self.module_index, self.module_pre_index, self.hptcool_inlet_mach_number, self.hpt_outlet_mach_number]
        qref = run_hptcool_core(args)


        """
        gamma, cp = self.cref[2, self.module_index], self.cref[1, self.module_index]

        gamma_t, gamma_p, gamma_w, rg = calc_coef(cp, gamma)

        # 流入マッハ数
        mc41 = self.hptcool_inlet_mach_number
        TT, PT = qref[4, self.module_pre_index], qref[5, self.module_pre_index]
        tau, pai = qref[7, self.module_pre_index], qref[8, self.module_pre_index]
        W = qref[1, self.module_pre_index]
        # 流量
        W41 = W
        TT41, PT41, MFP41, A41 = calc_thermal_indexes(W41, TT, PT, tau, pai, mc41, gamma, gamma_t, gamma_w, rg)

        # 圧力比・温度比を計算
        cp25 = self.cref[1, 25]
        TT30 = qref[4, 30]
        cbr40 = self.cref[4, 40]
        W20 = qref[1, 20]
        eps41 = self.cref[3, self.module_index]
        pai41 = 1.0 - eps41
        tau41 = 1.0 - (1.0 - cp25 * TT30 / (cp * TT41)) / (1.0 + W41 / (W20 * cbr40))

        # エントロピー計算
        L41 = 0.0

        qref[1, self.module_index] = W41
        qref[2, self.module_index] = A41
        qref[3, self.module_index] = 1.0  # 回転数
        qref[4, self.module_index] = TT41
        qref[5, self.module_index] = PT41
        qref[6, self.module_index] = L41  # エントロピー
        qref[7, self.module_index] = tau41
        qref[8, self.module_index] = pai41
        qref[9, self.module_index] = MFP41

        # 推力f00
        qref[0, self.module_index] = 0.0

        # 出口計算
        mc41e = self.hpt_outlet_mach_number
        TT41e, PT41e, MFP41e, A41e = calc_thermal_indexes(W41, TT41, PT41, tau41, pai41, mc41e, gamma, gamma_t, gamma_w,
                                                          rg)

        qref[11, self.module_index] = A41e
        """

        return qref


class LPT(InitEngineParamsCore):

    def __init__(self, cref, dref, propulsion_type):
        # 継承
        super().__init__()
        # パラメータが入っている行列
        self.cref = cref
        self.dref = dref
        # 推進形態(propulsion type)によってつながっているモジュールが異なる
        self.module_pre_index = 41  # (default)
        # モジュール番号
        self.module_index = 45

        self.propulsion_type = propulsion_type

    def __call__(self, qref, qref_e):

        args = [qref, qref_e, self.dref, self.cref, self.module_index, self.module_pre_index, self.lpt_inlet_mach_number, self.propulsion_type]
        qref = run_lpt_core(args)

        """
        gamma, cp = self.cref[2, self.module_index], self.cref[1, self.module_index]

        gamma_t, gamma_p, gamma_w, rg = calc_coef(cp, gamma)

        # 流入マッハ数
        mc45 = self.lpt_inlet_mach_number
        TT, PT = qref[4, self.module_pre_index], qref[5, self.module_pre_index]
        tau, pai = qref[7, self.module_pre_index], qref[8, self.module_pre_index]
        W = qref[1, self.module_pre_index]

        # 流量
        W20 = qref[1, 20]
        cbr40 = self.cref[4, 40]
        W45 = W + W20 * cbr40
        TT45, PT45, MFP45, A45 = calc_thermal_indexes(W45, TT, PT, tau, pai, mc45, gamma, gamma_t, gamma_w, rg)

        ytm45 = self.cref[5, self.module_index]
        ytd20 = self.cref[5, 20]
        # エネルギーマッチング(LP軸)

        # for electric
        ytaele = self.dref[34]  # electric change ratio
        if self.propulsion_type in ['turbojet', 'turboshaft', 'turbofan']:
            ytaele = 1.0

        LE10 = qref_e[6, 10]  # electric entarpy

        L10 = qref[6, 10]
        L20 = qref[6, 20]
        L45 = -(L10 + L20 + LE10 / ytaele) / (ytm45 + ytd20)

        ytp45 = self.cref[3, self.module_index]

        # 圧力比・温度比を計算
        tau45 = 1.0 + L45 / (W45 * cp * TT45)
        pai45 = tau45 ** (gamma_p / ytp45)

        ytt45 = (tau45 - 1.0) / (pai45 ** (1.0 / gamma_p) - 1.0)

        qref[1, self.module_index] = W45
        qref[2, self.module_index] = A45
        qref[3, self.module_index] = 1.0  # 回転数
        qref[4, self.module_index] = TT45
        qref[5, self.module_index] = PT45
        qref[6, self.module_index] = L45  # エントロピー
        qref[7, self.module_index] = tau45
        qref[8, self.module_index] = pai45
        qref[9, self.module_index] = MFP45

        qref[0, self.module_index] = ytt45

        # add electric target entarpy
        qref[10, 10] = LE10 / ytaele
        """

        return qref


class LPTCool(InitEngineParamsCore):

    def __init__(self, cref, dref, propulsion_type):
        # 継承
        super().__init__()
        # パラメータが入っている行列
        self.cref = cref
        self.dref = dref
        self.module_pre_index = 45  # (default)
        # モジュール番号
        self.module_index = 46

    def __call__(self, qref, qref_e):

        args = [qref, qref_e, self.dref, self.cref, self.module_index, self.module_pre_index, self.lptcool_inlet_mach_number, self.lpt_outlet_mach_number]
        qref = run_lptcool_core(args)

        """
        gamma, cp = self.cref[2, self.module_index], self.cref[1, self.module_index]

        gamma_t, gamma_p, gamma_w, rg = calc_coef(cp, gamma)

        # 流入マッハ数
        mc46 = self.lptcool_inlet_mach_number
        TT, PT = qref[4, self.module_pre_index], qref[5, self.module_pre_index]
        tau, pai = qref[7, self.module_pre_index], qref[8, self.module_pre_index]
        W = qref[1, self.module_pre_index]
        # 流量
        W46 = W
        TT46, PT46, MFP46, A46 = calc_thermal_indexes(W46, TT, PT, tau, pai, mc46, gamma, gamma_t, gamma_w, rg)

        # 圧力比・温度比を計算
        cp25 = self.cref[1, 25]
        TT30 = qref[4, 30]
        cbr45 = self.cref[4, 45]
        W20 = qref[1, 20]
        eps46 = self.cref[3, self.module_index]
        pai46 = 1.0 - eps46
        tau46 = 1.0 - (1.0 - cp25 * TT30 / (cp * TT46)) / (1.0 + W46 / (W20 * cbr45))
        # エントロピー計算
        L46 = 0.0

        qref[1, self.module_index] = W46
        qref[2, self.module_index] = A46
        qref[3, self.module_index] = 1.0  # 回転数
        qref[4, self.module_index] = TT46
        qref[5, self.module_index] = PT46
        qref[6, self.module_index] = L46  # エントロピー
        qref[7, self.module_index] = tau46
        qref[8, self.module_index] = pai46
        qref[9, self.module_index] = MFP46

        # 推力f00
        qref[0, self.module_index] = 0.0

        # 出口計算
        mc46e = self.lpt_outlet_mach_number
        TT46e, PT46e, MFP46e, A46e = calc_thermal_indexes(W46, TT46, PT46, tau46, pai46, mc46e, gamma, gamma_t, gamma_w,
                                                          rg)

        qref[11, self.module_index] = A46e
        """

        return qref


class CoreOut(InitEngineParamsCore):

    def __init__(self, cref, dref, propulsion_type):
        # 継承
        super().__init__()
        # パラメータが入っている行列
        self.cref = cref
        self.dref = dref
        # 推進形態(propulsion type)によってつながっているモジュールが異なる
        self.module_pre_index = 46  # (default)
        # モジュール番号
        self.module_index = 50

    def __call__(self, qref, qref_e):

        args = [qref, qref_e, self.dref, self.cref, self.module_index, self.module_pre_index, self.nozzle_inlet_mach_number]
        qref = run_coreout_core(args)

        """
        gamma, cp = self.cref[2, self.module_index], self.cref[1, self.module_index]

        gamma_t, gamma_p, gamma_w, rg = calc_coef(cp, gamma)

        # 流入マッハ数
        mc50 = self.nozzle_inlet_mach_number
        TT, PT = qref[4, self.module_pre_index], qref[5, self.module_pre_index]
        tau, pai = qref[7, self.module_pre_index], qref[8, self.module_pre_index]
        W = qref[1, self.module_pre_index]

        # 流量
        W20 = qref[1, 20]
        cbr45 = self.cref[4, 45]
        W50 = W + W20 * cbr45
        TT50, PT50, MFP50, A50 = calc_thermal_indexes(W50, TT, PT, tau, pai, mc50, gamma, gamma_t, gamma_w, rg)

        # 圧力比・温度比を計算
        tau50 = 1.0
        pai50 = 1.0
        # エントロピー
        L50 = 0.0

        qref[1, self.module_index] = W50
        qref[2, self.module_index] = A50
        qref[3, self.module_index] = 1.0  # 回転数
        qref[4, self.module_index] = TT50
        qref[5, self.module_index] = PT50
        qref[6, self.module_index] = L50  # エントロピー
        qref[7, self.module_index] = tau50
        qref[8, self.module_index] = pai50
        qref[9, self.module_index] = MFP50

        qref[0, self.module_index] = 0.0
        """

        return qref


class AfterBurner(InitEngineParamsCore):

    def __init__(self, cref, dref, propulsion_type):
        # 継承
        super().__init__()
        # パラメータが入っている行列
        self.cref = cref
        self.dref = dref
        # 推進形態(propulsion type)によってつながっているモジュールが異なる
        self.module_pre_index = 50  # (default)
        # モジュール番号
        self.module_index = 70

    def __call__(self, qref, qref_e):

        args = [qref, qref_e, self.dref, self.cref, self.module_index, self.module_pre_index]
        qref = run_afterburner_core(args)

        """
        gamma, cp = self.cref[2, self.module_index], self.cref[1, self.module_index]

        gamma_t, gamma_p, gamma_w, rg = calc_coef(cp, gamma)

        # 流入マッハ数
        mc70 = 0.2
        TT, PT = qref[4, self.module_pre_index], qref[5, self.module_pre_index]
        tau, pai = qref[7, self.module_pre_index], qref[8, self.module_pre_index]
        W = qref[1, self.module_pre_index]

        # 流量
        W70 = W
        TT70, PT70, MFP70, A70 = calc_thermal_indexes(W70, TT, PT, tau, pai, mc70, gamma, gamma_t, gamma_w, rg)

        # 圧力比・温度比を計算
        eps70 = self.cref[3, self.module_index]
        hlv70 = self.cref[4, self.module_index]
        ytb70 = self.cref[5, self.module_index]
        tau70 = 1.0
        pai70 = 1.0 - eps70

        # エントロピー
        L70 = 0.0
        # 燃料消費量
        WF70 = W70 * cp * TT70 * (tau70 - 1.0) / (ytb70 * hlv70 - cp * TT70 * tau70)

        qref[1, self.module_index] = W70
        qref[2, self.module_index] = A70
        qref[3, self.module_index] = 0.0  # 回転数
        qref[4, self.module_index] = TT70
        qref[5, self.module_index] = PT70
        qref[6, self.module_index] = L70  # エントロピー
        qref[7, self.module_index] = tau70
        qref[8, self.module_index] = pai70
        qref[9, self.module_index] = MFP70

        qref[0, self.module_index] = WF70  # 燃料消費量
        """

        return qref


class Nozzle(InitEngineParamsCore):

    def __init__(self, cref, dref, propulsion_type):
        # 継承
        super().__init__()
        # パラメータが入っている行列
        self.cref = cref
        self.dref = dref
        # 推進形態(propulsion type)によってつながっているモジュールが異なる
        self.module_pre_index = 50  # (default)
        # モジュール番号
        self.module_index = 80

    def __call__(self, qref, qref_e):

        args = [qref, qref_e, self.dref, self.cref, self.module_index, self.module_pre_index, self.nozzle_inlet_mach_number]
        qref = run_nozzle_core(args)

        """
        gamma, cp = self.cref[2, self.module_index], self.cref[1, self.module_index]

        gamma_t, gamma_p, gamma_w, rg = calc_coef(cp, gamma)

        # 流入マッハ数
        mc80 = self.nozzle_inlet_mach_number
        TT, PT = qref[4, self.module_pre_index], qref[5, self.module_pre_index]
        tau, pai = qref[7, self.module_pre_index], qref[8, self.module_pre_index]
        W = qref[1, self.module_pre_index]
        WF70 = qref[0, self.module_pre_index]

        # 流量
        W80 = W + WF70
        TT80, PT80, MFP80, A80 = calc_thermal_indexes(W80, TT, PT, tau, pai, mc80, gamma, gamma_t, gamma_w, rg)

        # 圧力比・温度比を計算
        pai80 = self.cref[3, self.module_index]
        tau80 = 1.0

        # エントロピー
        L80 = 0.0

        qref[1, self.module_index] = W80
        qref[2, self.module_index] = A80
        qref[3, self.module_index] = 0.0  # 回転数
        qref[4, self.module_index] = TT80
        qref[5, self.module_index] = PT80
        qref[6, self.module_index] = L80  # エントロピー
        qref[7, self.module_index] = tau80
        qref[8, self.module_index] = pai80
        qref[9, self.module_index] = MFP80
        """

        return qref


class Jet(InitEngineParamsCore):

    def __init__(self, cref, dref, propulsion_type):
        # 継承
        super().__init__()
        # パラメータが入っている行列
        self.cref = cref
        self.dref = dref
        # 推進形態(propulsion type)によってつながっているモジュールが異なる
        self.module_pre_index = 80  # (default)
        # モジュール番号
        self.module_index = 90

    def __call__(self, qref, qref_e):

        args = [qref, qref_e, self.dref, self.cref, self.module_index, self.module_pre_index]
        qref = run_jet_core(args)

        """
        gamma, cp = self.cref[2, self.module_index], self.cref[1, self.module_index]

        gamma_t, gamma_p, gamma_w, rg = calc_coef(cp, gamma)

        # 上流性能
        TT, PT = qref[4, self.module_pre_index], qref[5, self.module_pre_index]
        tau, pai = qref[7, self.module_pre_index], qref[8, self.module_pre_index]
        W = qref[1, self.module_pre_index]

        # 流量
        W90 = W
        TT90 = TT * tau
        PT90 = PT * pai

        # 圧力比・温度比を求める
        pai90 = 1.0
        tau90 = 1.0

        # エントロピー
        L90 = 0.0

        ####################チョーク判定#########################
        pcr90 = PT90 / (1.0 + gamma_t) ** gamma_p
        P00 = self.dref[5]  # 静圧

        if P00 <= pcr90:
            mc90 = 1.0  # マッハ数
            P90 = pcr90

        else:
            mc90 = np.sqrt(((PT90 / P00) ** (1.0 / gamma_p) - 1.0) / gamma_t)
            P90 = P00

        MFP90 = np.sqrt(gamma / rg) * mc90 / (1.0 + gamma_t * mc90 ** 2) ** gamma_w
        A90 = W90 / (PT90 / np.sqrt(TT90) * MFP90)
        U90 = mc90 * np.sqrt(gamma * rg * TT90 / (1.0 + gamma_t))

        # convergence confirmination (for calcaulation is expanding)
        # print('W90:',W90,'U90:',U90,'A90:',A90,'P90:',P90,'P00:',P00,'mc90:',mc90)

        ######################################################

        qref[1, self.module_index] = W90
        qref[2, self.module_index] = A90
        qref[3, self.module_index] = 1.0  # 回転数
        qref[4, self.module_index] = TT90
        qref[5, self.module_index] = PT90
        qref[6, self.module_index] = L90  # エントロピー
        qref[7, self.module_index] = tau90
        qref[8, self.module_index] = pai90
        qref[9, self.module_index] = MFP90

        qref[0, self.module_index] = W90 * U90 + A90 * (P90 - P00)  # 推力f90
        qref[10, self.module_index] = U90
        """


        return qref


class FanNozzle(InitEngineParamsCore):

    def __init__(self, cref, dref, propulsion_type):
        # 継承
        super().__init__()
        # パラメータが入っている行列
        self.cref = cref
        self.dref = dref
        # 推進形態(propulsion type)によってつながっているモジュールが異なる
        self.module_pre_index = 10  # (default)
        # モジュール番号
        self.module_index = 18

    def __call__(self, qref, qref_e):

        args = [qref, qref_e, self.dref, self.cref, self.module_index, self.module_pre_index, self.nozzle_inlet_mach_number]
        qref = run_fannozzle_core(args)

        """
        gamma, cp = self.cref[2, self.module_index], self.cref[1, self.module_index]

        gamma_t, gamma_p, gamma_w, rg = calc_coef(cp, gamma)

        # 流入マッハ数
        mc18 = self.nozzle_inlet_mach_number
        TT, PT = qref[4, self.module_pre_index], qref[5, self.module_pre_index]
        tau, pai = qref[7, self.module_pre_index], qref[8, self.module_pre_index]

        # 流量
        BPR = self.dref[20]
        W18 = BPR
        TT18, PT18, MFP18, A18 = calc_thermal_indexes(W18, TT, PT, tau, pai, mc18, gamma, gamma_t, gamma_w, rg)

        # 圧力比・温度比を計算
        pai18 = self.cref[3, self.module_index]
        tau18 = 1.0

        # エントロピー
        L18 = 0.0

        qref[1, self.module_index] = W18
        qref[2, self.module_index] = A18
        qref[3, self.module_index] = 0.0  # 回転数
        qref[4, self.module_index] = TT18
        qref[5, self.module_index] = PT18
        qref[6, self.module_index] = L18  # エントロピー
        qref[7, self.module_index] = tau18
        qref[8, self.module_index] = pai18
        qref[9, self.module_index] = MFP18
        """

        return qref


class FanJet(InitEngineParamsCore):

    def __init__(self, cref, dref, propulsion_type):
        # 継承
        super().__init__()
        # パラメータが入っている行列
        self.cref = cref
        self.dref = dref
        # 推進形態(propulsion type)によってつながっているモジュールが異なる
        self.module_pre_index = 18  # (default)
        # モジュール番号
        self.module_index = 19

    def __call__(self, qref, qref_e):

        args = [qref, qref_e, self.dref, self.cref, self.module_index, self.module_pre_index]

        qref = run_fanjet_core(args)

        """
        gamma, cp = self.cref[2, self.module_index], self.cref[1, self.module_index]

        gamma_t, gamma_p, gamma_w, rg = calc_coef(cp, gamma)

        # 上流性能
        TT, PT = qref[4, self.module_pre_index], qref[5, self.module_pre_index]
        tau, pai = qref[7, self.module_pre_index], qref[8, self.module_pre_index]
        W = qref[1, self.module_pre_index]

        # 流量
        W19 = W
        TT19 = TT * tau
        PT19 = PT * pai

        # 圧力比・温度比を求める
        pai19 = 1.0
        tau19 = 1.0

        # エントロピー
        L19 = 0.0

        ####################チョーク判定#########################
        pcr19 = PT19 / (1.0 + gamma_t) ** gamma_p
        P00 = self.dref[5]  # 静圧

        if P00 <= pcr19:
            mc19 = 1.0  # マッハ数
            P19 = pcr19

        else:
            mc19 = np.sqrt(((PT19 / P00) ** (1.0 / gamma_p) - 1.0) / gamma_t)
            P19 = P00

        MFP19 = np.sqrt(gamma / rg) * mc19 / (1.0 + gamma_t * mc19 ** 2) ** gamma_w
        A19 = W19 / (PT19 / np.sqrt(TT19) * MFP19)
        U19 = mc19 * np.sqrt(gamma * rg * TT19 / (1.0 + gamma_t))

        ######################################################

        qref[1, self.module_index] = W19
        qref[2, self.module_index] = A19
        qref[3, self.module_index] = 0.0  # 回転数
        qref[4, self.module_index] = TT19
        qref[5, self.module_index] = PT19
        qref[6, self.module_index] = L19  # エントロピー
        qref[7, self.module_index] = tau19
        qref[8, self.module_index] = pai19
        qref[9, self.module_index] = MFP19

        qref[0, self.module_index] = W19 * U19 + A19 * (P19 - P00)  # 推力f19
        qref[10, self.module_index] = U19
        """

        return qref


#######################Electric Engine Component Module##############################
class InletElec(object):

    def __init__(self, cref_e, dref, propulsion_type):
        # パラメータが入っている行列
        self.cref_e = cref_e
        self.dref = dref
        # モジュール番号
        self.module_index = 0

    def __call__(self, qref_e):

        args = [qref_e, self.dref, self.cref_e, self.module_index]
        qref_e = run_inlet_elec(args)

        """
        gamma, cp = self.cref_e[2, self.module_index], self.cref_e[1, self.module_index]

        gamma_t, gamma_p, gamma_w, rg = calc_coef(cp, gamma)

        # 流入マッハ数
        mc00 = self.dref[1]
        T00, P00 = self.dref[4], self.dref[5]
        BPRe = self.dref[31]
        # 全温・全圧
        TT00 = T00 * (1.0 + gamma_t * mc00 ** 2)
        PT00 = P00 * (1.0 + gamma_t * mc00 ** 2) ** gamma_p
        # コア側流量　1として計算する
        W00 = BPRe
        MFP00 = np.sqrt(gamma / rg) * mc00 / (1.0 + gamma_t * mc00 ** 2) ** gamma_w
        # 面積
        A00 = W00 / (PT00 / np.sqrt(TT00) * max(MFP00, 1.0e-10))
        # 速度
        U00 = mc00 * np.sqrt(gamma * rg * TT00 / (1.0 + gamma_t * mc00 ** 2))

        # 圧力比・温度比を計算
        ytd00 = self.cref_e[3, self.module_index]
        pai00 = 1.0 / ((1.0 + (1.0 - ytd00) * gamma_t * mc00 ** 2) ** gamma_p)
        tau00 = 1.0

        qref_e[1, self.module_index] = W00
        qref_e[2, self.module_index] = A00
        qref_e[3, self.module_index] = 0.0  # 回転数
        qref_e[4, self.module_index] = TT00
        qref_e[5, self.module_index] = PT00
        qref_e[6, self.module_index] = 0.0  # エントロピー
        qref_e[7, self.module_index] = tau00
        qref_e[8, self.module_index] = pai00
        qref_e[9, self.module_index] = MFP00

        # 推力f00
        qref_e[0, self.module_index] = -W00 * U00
        """

        return qref_e


class FanElec(InitEngineParamsCore):

    def __init__(self, cref_e, dref, propulsion_type):
        # 継承
        super().__init__()
        # パラメータが入っている行列
        self.cref_e = cref_e
        self.dref = dref
        # モジュール番号
        self.module_index = 10

    def __call__(self, qref_e):

        args = [qref_e, self.dref, self.cref_e, self.module_index, self.fan_inlet_mach_number]
        qref_e = run_fan_elec(args)

        """
        gamma, cp = self.cref_e[2, self.module_index], self.cref_e[1, self.module_index]

        gamma_t, gamma_p, gamma_w, rg = calc_coef(cp, gamma)

        # 流入マッハ数
        mc10 = self.fan_inlet_mach_number
        TT00, PT00 = qref_e[4, 0], qref_e[5, 0]
        tau00, pai00 = qref_e[7, 0], qref_e[8, 0]
        W00 = qref_e[1, 0]

        # 流量
        W10 = W00

        TT10, PT10, MFP10, A10 = calc_thermal_indexes(W10, TT00, PT00, tau00, pai00, mc10, gamma, gamma_t, gamma_w, rg)

        # 圧力比・温度比を計算
        FPRe = self.dref[32]
        ytp10 = self.cref_e[3, self.module_index]
        pai10 = FPRe
        tau10 = pai10 ** (1.0 / (gamma_p * ytp10))

        # エントロピー計算
        L10 = W10 * cp * TT10 * (tau10 - 1.0)
        ytc10 = (pai10 ** (1.0 / gamma_p) - 1.0) / (tau10 - 1.0)

        qref_e[1, self.module_index] = W10
        qref_e[2, self.module_index] = A10
        qref_e[3, self.module_index] = 1.0  # 回転数
        qref_e[4, self.module_index] = TT10
        qref_e[5, self.module_index] = PT10
        qref_e[6, self.module_index] = L10  # エントロピー
        qref_e[7, self.module_index] = tau10
        qref_e[8, self.module_index] = pai10
        qref_e[9, self.module_index] = MFP10

        # 推力f00
        qref_e[0, self.module_index] = ytc10
        """

        return qref_e


class FanNozzleElec(InitEngineParamsCore):

    def __init__(self, cref_e, dref, propulsion_type):
        # 継承
        super().__init__()
        # パラメータが入っている行列
        self.cref_e = cref_e
        self.dref = dref
        # 推進形態(propulsion type)によってつながっているモジュールが異なる
        self.module_pre_index = 10  # (default)
        # モジュール番号
        self.module_index = 18

    def __call__(self, qref_e):

        args = [qref_e, self.dref, self.cref_e, self.module_index, self.module_pre_index, self.nozzle_inlet_mach_number]
        qref_e = run_fannozzle_elec(args)

        """
        gamma, cp = self.cref_e[2, self.module_index], self.cref_e[1, self.module_index]

        gamma_t, gamma_p, gamma_w, rg = calc_coef(cp, gamma)

        # 流入マッハ数
        mc18 = self.nozzle_inlet_mach_number
        TT, PT = qref_e[4, self.module_pre_index], qref_e[5, self.module_pre_index]
        tau, pai = qref_e[7, self.module_pre_index], qref_e[8, self.module_pre_index]

        # 流量
        W18 = qref_e[1, self.module_pre_index]
        TT18, PT18, MFP18, A18 = calc_thermal_indexes(W18, TT, PT, tau, pai, mc18, gamma, gamma_t, gamma_w, rg)

        # 圧力比・温度比を計算
        pai18 = self.cref_e[3, self.module_index]
        tau18 = 1.0

        # エントロピー
        L18 = 0.0

        qref_e[1, self.module_index] = W18
        qref_e[2, self.module_index] = A18
        qref_e[3, self.module_index] = 0.0  # 回転数
        qref_e[4, self.module_index] = TT18
        qref_e[5, self.module_index] = PT18
        qref_e[6, self.module_index] = L18  # エントロピー
        qref_e[7, self.module_index] = tau18
        qref_e[8, self.module_index] = pai18
        qref_e[9, self.module_index] = MFP18
        """

        return qref_e


class FanJetElec(InitEngineParamsCore):

    def __init__(self, cref_e, dref, propulsion_type):
        # 継承
        super().__init__()
        # パラメータが入っている行列
        self.cref_e = cref_e
        self.dref = dref
        self.propulsion_type = propulsion_type
        # 推進形態(propulsion type)によってつながっているモジュールが異なる
        self.module_pre_index = 18  # (default)
        # モジュール番号
        self.module_index = 19

    def __call__(self, qref):

        args = [qref, self.dref, self.cref_e, self.module_index, self.module_pre_index]
        qref = run_fanjet_elec(args)

        """
        gamma, cp = self.cref_e[2, self.module_index], self.cref_e[1, self.module_index]

        gamma_t, gamma_p, gamma_w, rg = calc_coef(cp, gamma)

        # 上流性能
        TT, PT = qref[4, self.module_pre_index], qref[5, self.module_pre_index]
        tau, pai = qref[7, self.module_pre_index], qref[8, self.module_pre_index]
        W = qref[1, self.module_pre_index]

        # 流量
        W19 = W
        TT19 = TT * tau
        PT19 = PT * pai

        # 圧力比・温度比を求める
        pai19 = 1.0
        tau19 = 1.0

        # エントロピー
        L19 = 0.0

        ####################チョーク判定#########################
        pcr19 = PT19 / (1.0 + gamma_t) ** gamma_p
        P00 = self.dref[5]  # 静圧

        if P00 <= pcr19:
            mc19 = 1.0  # マッハ数
            P19 = pcr19

        else:
            mc19 = np.sqrt(((PT19 / P00) ** (1.0 / gamma_p) - 1.0) / gamma_t)
            P19 = P00

        MFP19 = np.sqrt(gamma / rg) * mc19 / (1.0 + gamma_t * mc19 ** 2) ** gamma_w
        A19 = W19 / (PT19 / np.sqrt(TT19) * MFP19)
        U19 = mc19 * np.sqrt(gamma * rg * TT19 / (1.0 + gamma_t))

        # print('Electric Jet')
        # print('W19:',W19,'A19:',A19,'P19:',P19,'U19:',U19,'mc19:',mc19)

        ######################################################

        qref[1, self.module_index] = W19
        qref[2, self.module_index] = A19
        qref[3, self.module_index] = 0.0  # 回転数
        qref[4, self.module_index] = TT19
        qref[5, self.module_index] = PT19
        qref[6, self.module_index] = L19  # エントロピー
        qref[7, self.module_index] = tau19
        qref[8, self.module_index] = pai19
        qref[9, self.module_index] = MFP19

        qref[0, self.module_index] = W19 * U19 + A19 * (P19 - P00)  # 推力f19
        qref[10, self.module_index] = U19
        """

        return qref
