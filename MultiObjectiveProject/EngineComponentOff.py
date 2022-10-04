import numpy as np
from init_engine import InitEngineParamsCore
from EngineComponentOff_utils import *

"""
station number:{'Inlet':0,'Fan':10,'LPC':20,'HPC':25,'CC':30,'HPT':40,'HPTCool':41,'LPT':45,'LPTCool':46,'CoreOut':50,'AfterBurner':70,'Nozzle':80,'Jet':90}
"""

# 係数計算
def calc_coef(cp, gamma):
    gamma_t = 0.5 * (gamma - 1.0)
    gamma_p = gamma / (gamma - 1)
    gamma_w = 0.5 * (gamma + 1.0) / (gamma - 1.0)
    rg = cp / gamma_p

    return (gamma_t, gamma_p, gamma_w, rg)


# helper function

# 回転数をもとに設計点に対する温度比率・圧力比率を求める関数
# prmwl=rev_lp_rate,prmlp=rev_lp
def calc_tau_pai_ratio(prmwl, prmlp, alpha, beta, gamma_p, ytpref, pairef, tauref):
    prmt = np.abs(2.0 - (prmwl / prmlp) ** beta) ** (1.0 / alpha) * prmlp ** 2
    ltsq = (prmwl / prmlp - 1.0) ** 2.0 + (prmt / (prmlp ** 2) - 1.0) ** 2.0
    ytpoff = ytpref * np.exp(-ltsq) * np.exp(-(1.0 - prmlp) ** 2)
    pai = (1.0 + (pairef ** (1.0 / (ytpref * gamma_p)) - 1.0) * prmt) ** (ytpoff * gamma_p)
    ytc = (pai ** (1.0 / gamma_p) - 1.0) / (prmt * (tauref - 1.0))

    return prmt, pai, ytc


######################################Core part############################################

class Inlet(InitEngineParamsCore):

    def __init__(self, coff, doff, qref, propulsion_type):
        super().__init__()

        self.coff = coff.copy()
        self.doff = doff.copy()
        self.qref = qref.copy()

        self.propulsion_type = propulsion_type

        # current module station number
        self.module_index = 0
        # previous curent module station number
        self.module_pre_index = 0

    def __call__(self, qoff, qoff_e, rev_rate, rev_lp):
        """
        args:qoff(all results array)
             rev_rate:fixied airflow rate
             rev_lp:revolve rate(ex. 0.95)
        """

        args = [qoff, qoff_e, self.doff, self.coff, self.module_index]
        qoff = run_inlet_core(args)

        """
        gamma, cp = self.coff[2, 0], self.coff[1, 0]

        # calculate coefficients
        gamma_t, gamma_p, gamma_w, rg = calc_coef(cp, gamma)

        mach, T00, P00 = self.doff[1], self.doff[4], self.doff[5]

        mc00 = mach
        TT00 = T00 * (1.0 + gamma_t * mc00 ** 2)
        PT00 = P00 * (1.0 + gamma_t * mc00 ** 2) ** gamma_p
        MFP00 = np.sqrt(gamma / rg) * mc00 / (1.0 + gamma_t * mc00 ** 2) ** gamma_w
        U00 = mc00 * np.sqrt(gamma * rg * TT00 / (1.0 + gamma_t * mc00 ** 2))

        ytd00 = self.coff[3, 0]
        # pressure ratio
        pai00 = 1.0 / ((1.0 + (1.0 - ytd00) * gamma_t * mc00 ** 2) ** gamma_p)
        # temperature ratio
        tau00 = 1.0

        # results
        qoff[3, 0:1] = 0.0
        qoff[4, 0:1] = TT00
        qoff[5, 0:1] = PT00
        qoff[6, 0:1] = 0.0
        qoff[7, 0:1] = tau00
        qoff[8, 0:1] = pai00
        qoff[9, 0:1] = MFP00
        qoff[10, 0:1] = U00
        """

        return qoff, rev_rate, rev_lp


class Fan(InitEngineParamsCore):

    def __init__(self, coff, doff, qref, propulsion_type):
        super().__init__()

        self.coff = coff.copy()
        self.doff = doff.copy()
        self.qref = qref.copy()

        self.propulsion_type = propulsion_type

        # current module station number
        self.module_index = 10
        # previous curent module station number
        self.module_pre_index = 0

    def __call__(self, qoff, qoff_e, rev_rate, rev_lp):
        """
        args:qoff(all results array)
             rev_rate:fixied airflow rate
             rev_lp:revolve rate(ex. 0.95)
        """

        args = [qoff, qoff_e, self.qref, self.doff, self.coff, rev_rate, rev_lp, self.module_index, self.module_pre_index]
        qoff, rev_rate10, rev_lp10 = run_fan_core(args)

        """
        # Upper components performances
        U00 = qoff[10, self.module_pre_index]
        TT00 = qoff[4, self.module_pre_index]
        PT00 = qoff[5, self.module_pre_index]
        tau00 = qoff[7, self.module_pre_index]
        pai00 = qoff[8, self.module_pre_index]
        MFP00 = qoff[9, self.module_pre_index]

        # Design point performance
        A10ref = self.qref[2, self.module_index]
        ytp10ref = self.qref[3, self.module_index]
        tau10ref = self.qref[7, self.module_index]
        pai10ref = self.qref[8, self.module_index]
        MFP10ref = self.qref[9, self.module_index]

        gamma, cp = self.coff[2, self.module_index], self.coff[1, self.module_index]
        # calculate coefficients
        gamma_t, gamma_p, gamma_w, rg = calc_coef(cp, gamma)

        A10amp, paiamp = self.doff[42], self.doff[43]
        tauamp = paiamp ** (1.0 / gamma_p / ytp10ref)

        # calc indexes
        A10 = A10ref * A10amp
        TT10 = TT00 * tau00
        PT10 = PT00 * pai00

        # Prepare for calculating the operating point on the compressor map
        rev_lp10, rev_rate10 = rev_lp, rev_rate
        alpha, beta = self.coff[4, self.module_index], self.coff[5, self.module_index]

        # calculate required indexes for determining operating point on the compressor map
        prmt10, pai10, ytc10 = calc_tau_pai_ratio(rev_rate10, rev_lp10, alpha, beta, gamma_p, ytp10ref, pai10ref,
                                                  tau10ref)

        # calculate MFP Area Velocity
        MFP10 = MFP10ref * rev_rate10
        W10 = MFP10 * (PT10 * A10) / np.sqrt(TT10)
        # calculate temperature ratio
        tau10 = (tau10ref * tauamp - 1.0) * prmt10 + 1.0
        # calculate entarpy
        L10 = W10 * cp * TT10 * (tau10ref * tauamp - 1.0) * prmt10

    
        # print(TT10,PT10)
        # print(MFP10,MFP10ref)
        # print(rev_rate10,rev_lp10,prmt10,ytp10ref,pai10ref,A10ref,W10)
        # exit()
        

        # redefine Inlet parameters
        W00 = W10
        A00 = W00 / (PT00 / np.sqrt(TT00) * max(MFP00, 1.0e-10))

        # results
        # Inlet
        qoff[1, 0] = W00
        qoff[2, 0] = A00
        qoff[0, 0] = -W00 * U00

        # Fan
        qoff[1, self.module_index:self.module_index + 1] = W10
        qoff[2, self.module_index:self.module_index + 1] = A10
        qoff[3, self.module_index:self.module_index + 1] = rev_lp10
        qoff[4, self.module_index:self.module_index + 1] = TT10
        qoff[5, self.module_index:self.module_index + 1] = PT10
        qoff[6, self.module_index:self.module_index + 1] = L10
        qoff[7, self.module_index:self.module_index + 1] = tau10
        qoff[8, self.module_index:self.module_index + 1] = pai10
        qoff[9, self.module_index:self.module_index + 1] = MFP10
        qoff[0, self.module_index:self.module_index + 1] = ytc10
        """


        return qoff, rev_rate10, rev_lp10


class LPC(InitEngineParamsCore):

    def __init__(self, coff, doff, qref, propulsion_type):

        super().__init__()

        self.coff = coff.copy()
        self.doff = doff.copy()
        self.qref = qref.copy()

        self.propulsion_type = propulsion_type

        # current module station number
        self.module_index = 20
        # previous curent module station number

        if self.propulsion_type in ['turbojet', 'turboshaft', 'TeDP']:
            self.module_pre_index = 0
        elif self.propulsion_type in ['turbofan', 'PartialElectric']:
            self.module_pre_index = 10

    def __call__(self, qoff, qoff_e, rev_rate, rev_lp):

        """
        args:qoff(all results array)
             rev_rate:fixied airflow rate
             rev_lp:revolve rate(ex. 0.95)
        """

        if self.propulsion_type in ['turbojet', 'turboshaft', 'TeDP']:

            qoff, rev_rate, rev_lp = self.run_nofan(qoff, qoff_e, rev_rate, rev_lp)

        elif self.propulsion_type in ['turbofan', 'PartialElectric']:

            qoff, rev_rate, rev_lp = self.run_fan(qoff, qoff_e, rev_rate, rev_lp)

        return qoff, rev_rate, rev_lp

    def run_nofan(self, qoff, qoff_e, rev_rate, rev_lp):

        args = [qoff, self.qref, self.doff, self.coff, rev_rate, rev_lp, self.module_index, self.module_pre_index]
        qoff, rev_rate20, rev_lp20 = run_lpc_nofan(args)

        """
        # Upper components
        U00 = qoff[10, self.module_pre_index]
        TT00 = qoff[4, self.module_pre_index]
        PT00 = qoff[5, self.module_pre_index]
        tau00 = qoff[7, self.module_pre_index]
        pai00 = qoff[8, self.module_pre_index]
        MFP00 = qoff[9, self.module_pre_index]

        # Design point performances
        A20ref = self.qref[2, self.module_index]
        ytp20ref = self.qref[3, self.module_index]
        tau20ref = self.qref[7, self.module_index]
        pai20ref = self.qref[8, self.module_index]
        MFP20ref = self.qref[9, self.module_index]

        gamma, cp = self.coff[2, self.module_index], self.coff[1, self.module_index]
        # calculate coefficients
        gamma_t, gamma_p, gamma_w, rg = calc_coef(cp, gamma)

        # current components performances
        A20 = A20ref
        TT20 = TT00 * tau00
        PT20 = PT00 * pai00

        # Prepare for calculating the operating point on the compressor map
        rev_lp20 = rev_lp
        # fixied airflow
        rev_rate20 = rev_rate

        # print('revolve LPC:',rev_lp20,'revolve rate LPC:',rev_rate20)

        alpha, beta = self.coff[4, self.module_index], self.coff[5, self.module_index]

        # calculate required indexes for determining operating point on the compressor map
        prmt20, pai20, ytc20 = calc_tau_pai_ratio(rev_rate20, rev_lp20, alpha, beta, gamma_p, ytp20ref, pai20ref,
                                                  tau20ref)

        MFP20 = MFP20ref * rev_rate20
        # Airflow rate
        W20 = MFP20 * (PT20 * A20) / np.sqrt(TT20)
        # temperature ratio
        tau20 = (tau20ref - 1.0) * prmt20 + 1.0
        # entarpy
        L20 = W20 * cp * TT20 * (tau20ref - 1.0) * prmt20

        # results
        # Inlet
        W00 = W20
        A00 = W00 / (PT00 / np.sqrt(TT00) * max(MFP00, 1.0e-10))

        qoff[1, self.module_pre_index:self.module_pre_index + 1] = W00
        qoff[2, self.module_pre_index:self.module_pre_index + 1] = A00
        # thrust
        qoff[0, self.module_pre_index:self.module_pre_index + 1] = -W00 * U00

        # LPC
        qoff[1, self.module_index:self.module_index + 1] = W20
        qoff[2, self.module_index:self.module_index + 1] = A20
        qoff[3, self.module_index:self.module_index + 1] = rev_lp20
        qoff[4, self.module_index:self.module_index + 1] = TT20
        qoff[5, self.module_index:self.module_index + 1] = PT20
        qoff[6, self.module_index:self.module_index + 1] = L20
        qoff[7, self.module_index:self.module_index + 1] = tau20
        qoff[8, self.module_index:self.module_index + 1] = pai20
        qoff[9, self.module_index:self.module_index + 1] = MFP20

        qoff[0, self.module_index:self.module_index + 1] = ytc20
        """

        return qoff, rev_rate20, rev_lp20

    def run_fan(self, qoff, qoff_e, rev_rate, rev_lp):

        args = [qoff, self.qref, self.doff, self.coff, rev_lp, self.module_index, self.module_pre_index]

        qoff, rev_rate20, rev_lp20 = run_lpc_fan(args)

        """
        # Upper components performances
        W10 = qoff[1, self.module_pre_index]
        TT10 = qoff[4, self.module_pre_index]
        PT10 = qoff[5, self.module_pre_index]
        tau10 = qoff[7, self.module_pre_index]
        pai10 = qoff[8, self.module_pre_index]

        # Design point performances
        tau10ref = self.qref[7, self.module_pre_index]
        A20ref = self.qref[2, self.module_index]
        ytp20ref = self.qref[3, self.module_index]
        tau20ref = self.qref[7, self.module_index]
        pai20ref = self.qref[8, self.module_index]
        MFP20ref = self.qref[9, self.module_index]

        W18 = qoff[1, 18]

        gamma, cp = self.coff[2, self.module_index], self.coff[1, self.module_index]
        # calculate coefficients
        gamma_t, gamma_p, gamma_w, rg = calc_coef(cp, gamma)

        # current component's performances
        W20 = W10 - W18
        TT20 = TT10 * tau10
        PT20 = PT10 * pai10
        A20 = A20ref
        MFP20 = W20 * np.sqrt(TT20) / (PT20 * A20)

        # Prepare for calculating the operating point on the compressor map
        rev_lp20 = rev_lp * np.sqrt(tau10ref / tau10)
        # fixied airflow
        rev_rate20 = MFP20 / MFP20ref

        # print('revolve LPC:',rev_lp20,'revolve rate LPC:',rev_rate20)

        alpha, beta = self.coff[6, self.module_index], self.coff[7, self.module_index]

        # calculate required indexes for determining operating point on the compressor map
        prmt20, pai20, ytc20 = calc_tau_pai_ratio(rev_rate20, rev_lp20, alpha, beta, gamma_p, ytp20ref, pai20ref,
                                                  tau20ref)

        # temperature ratio
        tau20 = (tau20ref - 1.0) * prmt20 + 1.0
        L20 = W20 * cp * TT20 * (tau20ref - 1.0) * prmt20

        # results
        qoff[1, self.module_index:self.module_index + 1] = W20
        qoff[2, self.module_index:self.module_index + 1] = A20
        qoff[3, self.module_index:self.module_index + 1] = rev_lp20
        qoff[4, self.module_index:self.module_index + 1] = TT20
        qoff[5, self.module_index:self.module_index + 1] = PT20
        qoff[6, self.module_index:self.module_index + 1] = L20
        qoff[7, self.module_index:self.module_index + 1] = tau20
        qoff[8, self.module_index:self.module_index + 1] = pai20
        qoff[9, self.module_index:self.module_index + 1] = MFP20

        qoff[0, self.module_index:self.module_index + 1] = ytc20
        """

        return qoff, rev_rate20, rev_lp20


class HPC(InitEngineParamsCore):

    def __init__(self, coff, doff, qref, propulsion_type, isformer=True):

        super().__init__()

        self.coff = coff.copy()
        self.doff = doff.copy()
        self.qref = qref.copy()

        self.propulsion_type = propulsion_type

        # Former or latter Flag
        self.isformer = isformer

        # current module station number
        self.module_index = 25
        # previous curent module station number
        self.module_pre_index = 20

    def __call__(self, qoff, qoff_e, rev_rate, rev):

        """
        args:qoff(all results array)
             rev_rate:fixied airflow rate
             rev:revolve rate(ex. 0.95)
        """
        rev_hp = 1.0

        if self.isformer:
            qoff, rev_rate, _ = self.run_former(qoff, qoff_e, rev_rate, rev)
        else:
            qoff, rev_rate, rev_hp = self.run_latter(qoff, qoff_e, rev_rate, rev)

        return qoff, rev_rate, rev_hp

    # Fomrer components convergence part
    def run_former(self, qoff, qoff_e, rev_rate, rev_lp):

        args = [qoff, self.qref, self.module_index, self.module_pre_index]
        qoff, rev_rate25 = run_hpc_former(args)

        """
        # Upper components performances
        W20 = qoff[1, self.module_pre_index]
        TT20 = qoff[4, self.module_pre_index]
        PT20 = qoff[5, self.module_pre_index]
        tau20 = qoff[7, self.module_pre_index]
        pai20 = qoff[8, self.module_pre_index]

        # Design point performances
        A25ref = self.qref[2, self.module_index]
        MFP25ref = self.qref[9, self.module_index]

        # current component performances
        W25 = W20
        TT25 = TT20 * tau20
        PT25 = PT20 * pai20
        A25 = A25ref
        MFP25 = W25 * np.sqrt(TT25) / (PT25 * A25)

        # fixied airflow rate
        rev_rate25 = MFP25 / MFP25ref

        # print('Fixied airflow rate:',rev_rate25)
        """

        return qoff, rev_rate25, rev_lp

    # latter compoenents convergence part
    def run_latter(self, qoff, qoff_e, rev_rate, rev_hp):
        args = [qoff, self.qref, self.doff, self.coff, rev_rate, rev_hp, self.module_index, self.module_pre_index, self.propulsion_type]
        qoff, rev_rate, rev_hp = run_hpc_latter(args)

        """
        # Upper components performances
        W20 = qoff[1, self.module_pre_index]
        TT20 = qoff[4, self.module_pre_index]
        PT20 = qoff[5, self.module_pre_index]
        tau20 = qoff[7, self.module_pre_index]
        pai20 = qoff[8, self.module_pre_index]

        # Design point performances
        A25ref = self.qref[2, self.module_index]
        ytp25ref = self.qref[3, self.module_index]
        tau25ref = self.qref[7, self.module_index]
        pai25ref = self.qref[8, self.module_index]
        MFP25ref = self.qref[9, self.module_index]

        gamma, cp = self.coff[2, self.module_index], self.coff[1, self.module_index]
        # calculate coefficients
        gamma_t, gamma_p, gamma_w, rg = calc_coef(cp, gamma)

        # current component performances
        W25 = W20
        TT25 = TT20 * tau20
        PT25 = PT20 * pai20
        A25 = A25ref
        MFP25 = W25 * np.sqrt(TT25) / (PT25 * A25)

        # fixied airflow rate
        rev_hp25 = rev_hp
        rev_rate25 = MFP25 / MFP25ref
        # Initialize default value(5,5)
        alpha, beta = 5, 5

        if self.propulsion_type in ['turbojet', 'turboshaft', 'TeDP']:

            alpha, beta = self.coff[6, self.module_index], self.coff[7, self.module_index]

        elif self.propulsion_type in ['turbofan', 'PartialElectric']:

            alpha, beta = self.coff[4, self.module_index], self.coff[5, self.module_index]

        # calculate required indexes for determining operating point on the compressor map
        prmt25, pai25, ytc25 = calc_tau_pai_ratio(rev_rate25, rev_hp25, alpha, beta, gamma_p, ytp25ref, pai25ref,
                                                  tau25ref)

        # temperature ratio
        tau25 = (tau25ref - 1.0) * prmt25 + 1.0
        L25 = W25 * cp * TT25 * (tau25ref - 1.0) * prmt25

        
        # print('')
        # print('tau25ref,pai25ref,tau25,pai25')
        # print(tau25ref,pai25ref,tau25,pai25)
        # print('Fixied airflow rate:',rev_rate25,'revolve hp shaft:',rev_hp25)

        # print('')

        # results
        qoff[1, self.module_index:self.module_index + 1] = W25
        qoff[2, self.module_index:self.module_index + 1] = A25
        qoff[3, self.module_index:self.module_index + 1] = rev_hp25
        qoff[4, self.module_index:self.module_index + 1] = TT25
        qoff[5, self.module_index:self.module_index + 1] = PT25
        qoff[6, self.module_index:self.module_index + 1] = L25
        qoff[7, self.module_index:self.module_index + 1] = tau25
        qoff[8, self.module_index:self.module_index + 1] = pai25
        qoff[9, self.module_index:self.module_index + 1] = MFP25

        qoff[0, self.module_index:self.module_index + 1] = ytc25
        """


        return qoff, rev_rate, rev_hp


class CC(InitEngineParamsCore):

    def __init__(self, coff, doff, qref, propulsion_type):
        super().__init__()

        self.coff = coff.copy()
        self.doff = doff.copy()
        self.qref = qref.copy()

        self.propulsion_type = propulsion_type

        # current module station number
        self.module_index = 30
        # previous curent module station number
        self.module_pre_index = 25

    def __call__(self, qoff, qoff_e, rev_rate, rev_hp):
        """
        args:qoff(all results array)
             rev_rate:fixied airflow rate
             rev_lp:revolve rate(ex. 0.95)
        """
        args = [qoff, qoff_e, self.qref, self.doff, self.coff, self.module_index, self.module_pre_index]
        qoff = run_cc_core(args)

        """
        # Upper components
        #####all previous airflow rate#####
        W20 = qoff[1, 20]
        W25 = qoff[1, self.module_pre_index]
        ###################################
        TT25 = qoff[4, self.module_pre_index]
        PT25 = qoff[5, self.module_pre_index]
        tau25 = qoff[7, self.module_pre_index]
        pai25 = qoff[8, self.module_pre_index]

        # Cool rate
        cbr40 = self.coff[4, 40]  # for hpt
        cbr45 = self.coff[4, 45]  # for lpt

        # Design point performances
        A30ref = self.qref[2, self.module_index]

        # when the performances of Combustion Chamber are calculated, we assume the choke of the entrance of HPT
        # We need the HPT's performance at design point
        A40ref = self.qref[2, 40]
        MFP40ref = self.qref[9, 40]

        gamma, cp = self.coff[2, self.module_index], self.coff[1, self.module_index]
        # calculate coefficients
        gamma_t, gamma_p, gamma_w, rg = calc_coef(cp, gamma)

        # Current compoenent performances
        W30 = W25 - (cbr40 + cbr45) * W20  # Subtract airflow for cooling turbines
        TT30 = TT25 * tau25
        PT30 = PT25 * pai25
        A30 = A30ref
        MFP30 = W30 * np.sqrt(TT30) / (PT30 * A30)

        eps30 = self.coff[3, self.module_index]  # lack of propulsive pressure
        hlv30 = self.coff[4, self.module_index]  # Lower heat volume
        ytb30 = self.coff[5, self.module_index]  # propulsive efficiency

        pai30 = 1.0 - eps30

        # Assume the High Pressure Turbone performances
        MFP40 = MFP40ref
        A40 = A40ref

        # outresults the temperature ratio of combustion chamber by solving 2 dimensional equations
        # prepare for calculating mathing of airflow
        # coeffficients
        cob30 = 0.5 * (ytb30 * hlv30 / (cp * TT30) - 1.0) * (MFP30 * A30 / (MFP40 * A40)) / pai30
        coc30 = -ytb30 * hlv30 / (cp * TT30)

        # solutions
        tau30 = (-cob30 + np.sqrt(cob30 ** 2 - coc30)) ** 2.0

        # print(cob30,coc30)

        # for SFC (fuel consumption)
        WF30 = W30 * cp * TT30 * (tau30 - 1.0) / (ytb30 * hlv30 - cp * TT30 * tau30)

        # entarpy
        L30 = ytb30 * WF30 * hlv30

        # print('30:',W30,TT30,PT30,tau30,pai30)

        # results
        qoff[1, self.module_index:self.module_index + 1] = W30
        qoff[2, self.module_index:self.module_index + 1] = A30
        qoff[3, self.module_index:self.module_index + 1] = 0.0  # not revolve
        qoff[4, self.module_index:self.module_index + 1] = TT30
        qoff[5, self.module_index:self.module_index + 1] = PT30
        qoff[6, self.module_index:self.module_index + 1] = L30
        qoff[7, self.module_index:self.module_index + 1] = tau30
        qoff[8, self.module_index:self.module_index + 1] = pai30
        qoff[9, self.module_index:self.module_index + 1] = MFP30

        # Fuel Consumption
        qoff[0, self.module_index:self.module_index + 1] = WF30
        """

        return qoff, rev_rate, rev_hp


class HPT(InitEngineParamsCore):

    def __init__(self, coff, doff, qref, propulsion_type):
        super().__init__()

        self.coff = coff.copy()
        self.doff = doff.copy()
        self.qref = qref.copy()

        self.propulsion_type = propulsion_type

        # current module station number
        self.module_index = 40
        # previous curent module station number
        self.module_pre_index = 30

    def __call__(self, qoff, qoff_e, rev_rate, rev_hp):
        """
        args:qoff(all results array)
             rev_rate:fixied airflow rate
             rev_lp:revolve rate(ex. 0.95)
        """
        args = [qoff, self.qref, self.doff, self.coff, rev_hp, self.module_index, self.module_pre_index]
        qoff, rev_hp40 = run_hpt_core(args)

        """
        # Upper components performances
        L25 = qoff[6, 25]  # HPC entarpy(for enegy matting at hp shaft)
        W30 = qoff[1, self.module_pre_index]
        TT30 = qoff[4, self.module_pre_index]
        PT30 = qoff[5, self.module_pre_index]
        tau30 = qoff[7, self.module_pre_index]
        pai30 = qoff[8, self.module_pre_index]
        WF30 = qoff[0, self.module_pre_index]  # Fuel consumption

        # Design point performances
        A40ref = self.qref[2, self.module_index]
        tau40ref = self.qref[7, self.module_index]
        pai40ref = self.qref[8, self.module_index]
        MFP40ref = self.qref[9, self.module_index]

        gamma, cp = self.coff[2, self.module_index], self.coff[1, self.module_index]
        # calculate coefficients
        gamma_t, gamma_p, gamma_w, rg = calc_coef(cp, gamma)

        # Current components performances
        W40 = W30 + WF30
        rev_hp40 = rev_hp
        A40 = A40ref
        TT40 = TT30 * tau30
        PT40 = PT30 * pai30
        MFP40 = W40 * np.sqrt(TT40) / (PT40 * A40)

        ytp40 = self.coff[3, self.module_index]
        tau40 = tau40ref  # assumption of HPT choking
        pai40 = tau40 ** (gamma_p / ytp40)
        ytt40 = (tau40 - 1.0) / (pai40 ** (1.0 / gamma_p) - 1.0)
        L40 = W40 * cp * TT40 * (tau40 - 1.0)

        # print('40:',PT40,TT40)

        # results
        qoff[1, self.module_index:self.module_index + 1] = W40
        qoff[2, self.module_index:self.module_index + 1] = A40
        qoff[3, self.module_index:self.module_index + 1] = rev_hp40
        qoff[4, self.module_index:self.module_index + 1] = TT40
        qoff[5, self.module_index:self.module_index + 1] = PT40
        qoff[6, self.module_index:self.module_index + 1] = L40
        qoff[7, self.module_index:self.module_index + 1] = tau40
        qoff[8, self.module_index:self.module_index + 1] = pai40
        qoff[9, self.module_index:self.module_index + 1] = MFP40

        qoff[0, self.module_index:self.module_index + 1] = ytt40
        """

        return qoff, rev_rate, rev_hp40


class HPTCool(InitEngineParamsCore):

    def __init__(self, coff, doff, qref, propulsion_type):
        super().__init__()

        self.coff = coff.copy()
        self.doff = doff.copy()
        self.qref = qref.copy()

        self.propulsion_type = propulsion_type

        # current module station number
        self.module_index = 41
        # previous curent module station number
        self.module_pre_index = 40

    def __call__(self, qoff, qoff_e):
        """
        args:qoff(all results array)
        """

        args = [qoff, self.qref, self.doff, self.coff, self.module_index, self.module_pre_index]
        qoff = run_hptcool_core(args)

        """
        # Upper component's performances
        W20 = qoff[1, 20]
        cp25 = self.coff[1, 25]
        TT30 = qoff[4, 30]
        W40 = qoff[1, self.module_pre_index]
        TT40 = qoff[4, self.module_pre_index]
        PT40 = qoff[5, self.module_pre_index]
        tau40 = qoff[7, self.module_pre_index]
        pai40 = qoff[8, self.module_pre_index]
        cbr40 = self.coff[4, self.module_pre_index]

        # Design point performances
        A41ref = self.qref[2, self.module_index]

        gamma, cp = self.coff[2, self.module_index], self.coff[1, self.module_index]
        # calculate coefficients
        gamma_t, gamma_p, gamma_w, rg = calc_coef(cp, gamma)

        # Current component performances
        W41 = W40
        TT41 = TT40 * tau40
        PT41 = PT40 * pai40
        A41 = A41ref
        MFP41 = W41 * np.sqrt(TT41) / (PT41 * A41)

        # temperature ratio and pressure ratio
        eps41 = self.coff[3, self.module_index]
        tau41 = 1.0 - (1.0 - cp25 * TT30 / (cp * TT41)) / (1.0 + W41 / (W20 * cbr40))
        pai41 = 1.0 - eps41
        L41 = 0.0

        # results
        qoff[1, self.module_index:self.module_index + 1] = W41
        qoff[2, self.module_index:self.module_index + 1] = A41
        qoff[3, self.module_index:self.module_index + 1] = 1.0
        qoff[4, self.module_index:self.module_index + 1] = TT41
        qoff[5, self.module_index:self.module_index + 1] = PT41
        qoff[6, self.module_index:self.module_index + 1] = L41
        qoff[7, self.module_index:self.module_index + 1] = tau41
        qoff[8, self.module_index:self.module_index + 1] = pai41
        qoff[9, self.module_index:self.module_index + 1] = MFP41

        qoff[0, self.module_index:self.module_index + 1] = 0.0
        """

        return qoff


class LPT(InitEngineParamsCore):

    def __init__(self, coff, doff, qref, propulsion_type):

        super().__init__()

        self.coff = coff.copy()
        self.doff = doff.copy()
        self.qref = qref.copy()

        self.propulsion_type = propulsion_type

        # current module station number
        self.module_index = 45
        # previous curent module station number
        self.module_pre_index = 41

    def __call__(self, qoff, qoff_e):

        """
        args:qoff(all results array)
        """

        args = [qoff, qoff_e, self.qref, self.doff, self.coff, self.module_index, self.module_pre_index, self.propulsion_type]
        qoff = run_lpt_core(args)

        """
        # Upper components
        W20 = qoff[1, 20]
        L10 = qoff[6, 10]  # Core fan entarpy
        L20 = qoff[6, 20]  # Core LPC entarpy
        cbr40 = self.coff[4, 40]

        W41 = qoff[1, self.module_pre_index]
        TT41 = qoff[4, self.module_pre_index]
        PT41 = qoff[5, self.module_pre_index]
        tau41 = qoff[7, self.module_pre_index]
        pai41 = qoff[8, self.module_pre_index]

        # Design point performances
        A45ref = self.qref[2, self.module_index]

        gamma, cp = self.coff[2, self.module_index], self.coff[1, self.module_index]
        # calculate coefficients
        gamma_t, gamma_p, gamma_w, rg = calc_coef(cp, gamma)

        # Current component performances
        W45 = W41 + cbr40 * W20  # add cool airflow rate
        TT45 = TT41 * tau41
        PT45 = PT41 * pai41
        A45 = A45ref
        MFP45 = W45 * np.sqrt(TT45) / (PT45 * A45)

        # Prepare for Entarpy distribution
        ytm45 = self.coff[5, self.module_index]
        ytadl = self.coff[4, 20]

        ytd20 = self.doff[50]

        ytaele = self.doff[34]  # electric efficiency

        if self.propulsion_type in ['turbojet', 'turboshaft', 'turbofan']:
            ytaele = 1.0

        LE10 = qoff_e[6, 10]  # electric entarpy

        # Entarpy Distribution

        L45 = -(LE10 / ytaele + L10 + L20) / (ytm45 + ytd20)

        # temperature ratio and pressure ratio
        ytp45 = self.coff[3, self.module_index]
        tau45 = 1.0 + L45 / (W45 * cp * TT45)
        pai45 = tau45 ** (gamma_p / ytp45)

        ytt45 = (tau45 - 1.0) / (pai45 ** (1.0 / gamma_p) - 1.0)

        # results
        qoff[1, self.module_index:self.module_index + 1] = W45
        qoff[2, self.module_index:self.module_index + 1] = A45
        qoff[3, self.module_index:self.module_index + 1] = 1.0
        qoff[4, self.module_index:self.module_index + 1] = TT45
        qoff[5, self.module_index:self.module_index + 1] = PT45
        qoff[6, self.module_index:self.module_index + 1] = L45
        qoff[7, self.module_index:self.module_index + 1] = tau45
        qoff[8, self.module_index:self.module_index + 1] = pai45
        qoff[9, self.module_index:self.module_index + 1] = MFP45

        qoff[0, self.module_index:self.module_index + 1] = ytt45

        # LE10
        qoff[10, 10:11] = LE10 / ytaele
        """

        return qoff


class LPTCool(InitEngineParamsCore):

    def __init__(self, coff, doff, qref, propulsion_type):
        super().__init__()

        self.coff = coff.copy()
        self.doff = doff.copy()
        self.qref = qref.copy()

        self.propulsion_type = propulsion_type

        # current module station number
        self.module_index = 46
        # previous curent module station number
        self.module_pre_index = 45

    def __call__(self, qoff, qoff_e):
        """
        args:qoff(all results array)
        """
        args = [qoff, self.qref, self.doff, self.coff, self.module_index, self.module_pre_index]
        qoff = run_lptcool_core(args)

        """
        # Upper component performances
        W20 = qoff[1, 20]
        cp25 = self.coff[1, 25]
        TT30 = qoff[4, 30]

        W45 = qoff[1, self.module_pre_index]
        TT45 = qoff[4, self.module_pre_index]
        PT45 = qoff[5, self.module_pre_index]
        tau45 = qoff[7, self.module_pre_index]
        pai45 = qoff[8, self.module_pre_index]
        cbr45 = self.coff[4, self.module_pre_index]

        # Design point performances
        A46ref = self.qref[2, self.module_index]

        gamma, cp = self.coff[2, self.module_index], self.coff[1, self.module_index]
        # calculate coefficients
        gamma_t, gamma_p, gamma_w, rg = calc_coef(cp, gamma)

        # Current component performances
        W46 = W45
        TT46 = TT45 * tau45
        PT46 = PT45 * pai45
        A46 = A46ref
        MFP46 = W46 * np.sqrt(TT46) / (PT46 * A46)

        # temperature ratio and pressure ratio
        eps46 = self.coff[3, self.module_index]
        tau46 = 1.0 - (1.0 - cp25 * TT30 / (cp * TT46)) / (1.0 + W46 / (W20 * cbr45))
        pai46 = 1.0 - eps46
        # print('tau46:', tau46, 'pai46:', pai46, 'TT46:', TT46, 'PT46:', PT46)
        L46 = 0.0

        # results
        qoff[1, self.module_index:self.module_index + 1] = W46
        qoff[2, self.module_index:self.module_index + 1] = A46
        qoff[3, self.module_index:self.module_index + 1] = 1.0
        qoff[4, self.module_index:self.module_index + 1] = TT46
        qoff[5, self.module_index:self.module_index + 1] = PT46
        qoff[6, self.module_index:self.module_index + 1] = L46
        qoff[7, self.module_index:self.module_index + 1] = tau46
        qoff[8, self.module_index:self.module_index + 1] = pai46
        qoff[9, self.module_index:self.module_index + 1] = MFP46

        qoff[0, self.module_index:self.module_index + 1] = 0.0
        """

        return qoff


class CoreOut(InitEngineParamsCore):

    def __init__(self, coff, doff, qref, propulsion_type):
        super().__init__()

        self.coff = coff.copy()
        self.doff = doff.copy()
        self.qref = qref.copy()

        self.propulsion_type = propulsion_type

        # current module station number
        self.module_index = 50
        # previous curent module station number
        self.module_pre_index = 46

    def __call__(self, qoff, qoff_e):
        """
        args:qoff(all results array)
        """

        args = [qoff, self.qref, self.doff, self.coff, self.module_index, self.module_pre_index]
        qoff = run_coreout_core(args)

        """
        # Upper component's performances
        W20 = qoff[1, 20]
        W46 = qoff[1, self.module_pre_index]
        TT46 = qoff[4, self.module_pre_index]
        PT46 = qoff[5, self.module_pre_index]
        tau46 = qoff[7, self.module_pre_index]
        pai46 = qoff[8, self.module_pre_index]
        cbr45 = self.coff[4, 45]

        gamma, cp = self.coff[2, self.module_index], self.coff[1, self.module_index]
        # calculate coefficients
        gamma_t, gamma_p, gamma_w, rg = calc_coef(cp, gamma)

        # Design point params
        A50ref = self.qref[2, self.module_index]

        W50 = W46 + W20 * cbr45
        TT50 = TT46 * tau46
        PT50 = PT46 * pai46
        A50 = A50ref
        MFP50 = W50 * np.sqrt(TT50) / (PT50 * A50)

        # temperature ratio and pressure ratio
        tau50 = 1.0
        pai50 = 1.0
        L50 = 0.0

        # results
        qoff[1, self.module_index:self.module_index + 1] = W50
        qoff[2, self.module_index:self.module_index + 1] = A50
        qoff[3, self.module_index:self.module_index + 1] = 1.0
        qoff[4, self.module_index:self.module_index + 1] = TT50
        qoff[5, self.module_index:self.module_index + 1] = PT50
        qoff[6, self.module_index:self.module_index + 1] = L50
        qoff[7, self.module_index:self.module_index + 1] = tau50
        qoff[8, self.module_index:self.module_index + 1] = pai50
        qoff[9, self.module_index:self.module_index + 1] = MFP50

        qoff[0, self.module_index:self.module_index + 1] = 0.0
        """

        return qoff


class AfterBurner(InitEngineParamsCore):

    def __init__(self, coff, doff, qref, propulsion_type):
        super().__init__()

        self.coff = coff.copy()
        self.doff = doff.copy()
        self.qref = qref.copy()

        self.propulsion_type = propulsion_type

        # current module station number
        self.module_index = 70
        # previous curent module station number
        self.module_pre_index = 50

        # Maximum Overall Temperatur
        self.TAB = 1000

    def __call__(self, qoff, qoff_e):
        """
        args:qoff(all results array)
        """

        args = [qoff, self.qref, self.doff, self.coff, self.module_index, self.module_pre_index]
        qoff = run_afterburner_core(args)

        """
        # Upper component's performances
        W50 = qoff[1, self.module_pre_index]
        TT50 = qoff[4, self.module_pre_index]
        PT50 = qoff[5, self.module_pre_index]
        tau50 = qoff[7, self.module_pre_index]
        pai50 = qoff[8, self.module_pre_index]

        # Design point performances
        A70ref = self.qref[2, self.module_index]

        gamma, cp = self.coff[2, self.module_index], self.coff[1, self.module_index]
        # calculate coefficients
        gamma_t, gamma_p, gamma_w, rg = calc_coef(cp, gamma)

        W70 = W50
        TT70 = TT50 * tau50
        PT70 = PT50 * pai50
        A70 = A70ref
        MFP70 = W70 * np.sqrt(TT70) / (PT70 * A70)

        eps70 = self.coff[3, self.module_index]
        hlv70 = self.coff[4, self.module_index]
        ytb70 = self.coff[5, self.module_index]

        # temperature ratio and pressure ratio
        pai70 = 1.0 - eps70
        tau70 = max(self.TAB, TT70) / TT70
        L70 = 0.0
        # Fuel consumption
        WF70 = W70 * cp * TT70 * (tau70 - 1.0) / (ytb70 * hlv70 - cp * TT70 * tau70)

        # results
        qoff[1, self.module_index:self.module_index + 1] = W70
        qoff[2, self.module_index:self.module_index + 1] = A70
        qoff[3, self.module_index:self.module_index + 1] = 0.0  # No revolve
        qoff[4, self.module_index:self.module_index + 1] = TT70
        qoff[5, self.module_index:self.module_index + 1] = PT70
        qoff[6, self.module_index:self.module_index + 1] = L70
        qoff[7, self.module_index:self.module_index + 1] = tau70
        qoff[8, self.module_index:self.module_index + 1] = pai70
        qoff[9, self.module_index:self.module_index + 1] = MFP70

        qoff[0, self.module_index:self.module_index + 1] = WF70
        """

        return qoff


class Nozzle(InitEngineParamsCore):

    def __init__(self, coff, doff, qref, propulsion_type):
        super().__init__()

        self.coff = coff.copy()
        self.doff = doff.copy()
        self.qref = qref.copy()

        self.propulsion_type = propulsion_type

        # current module station number
        self.module_index = 80
        # previous curent module station number
        # self.module_pre_index = 70#if afterburner in
        self.module_pre_index = 50

    def __call__(self, qoff, qoff_e):
        """
        args:qoff(all results array)
        """

        args = [qoff, self.qref, self.doff, self.coff, self.module_index, self.module_pre_index]
        qoff = run_nozzle_core(args)

        """
        # Upper components
        W50 = qoff[1, self.module_pre_index]
        TT50 = qoff[4, self.module_pre_index]
        PT50 = qoff[5, self.module_pre_index]
        tau50 = qoff[7, self.module_pre_index]
        pai50 = qoff[8, self.module_pre_index]

        # Design point params
        A80ref = self.qref[2, self.module_index]

        # if total propulsive system includes afterburner
        # WF70 = qoff[0,self.module_pre_index]
        # W80 = W70 + WF70

        W80 = W50
        TT80 = TT50 * tau50
        PT80 = PT50 * pai50
        A80 = A80ref
        MFP80 = W80 * np.sqrt(TT80) / (PT80 * A80)

        # temperature ratio and pressure ratio
        tau80 = 1.0
        pai80 = self.coff[3, self.module_index]
        L80 = 0.0

        # results
        qoff[1, self.module_index:self.module_index + 1] = W80
        qoff[2, self.module_index:self.module_index + 1] = A80
        qoff[3, self.module_index:self.module_index + 1] = 0.0  # No revolve
        qoff[4, self.module_index:self.module_index + 1] = TT80
        qoff[5, self.module_index:self.module_index + 1] = PT80
        qoff[6, self.module_index:self.module_index + 1] = L80
        qoff[7, self.module_index:self.module_index + 1] = tau80
        qoff[8, self.module_index:self.module_index + 1] = pai80
        qoff[9, self.module_index:self.module_index + 1] = MFP80

        qoff[0, self.module_index:self.module_index + 1] = 0.0
        """

        return qoff


class Jet(InitEngineParamsCore):

    def __init__(self, coff, doff, qref, propulsion_type):

        super().__init__()

        self.coff = coff.copy()
        self.doff = doff.copy()
        self.qref = qref.copy()

        self.propulsion_type = propulsion_type

        # current module station number
        self.module_index = 90
        # previous curent module station number
        self.module_pre_index = 80

    def __call__(self, qoff, qoff_e):

        """
        args:qoff(all results array)
        """

        args = [qoff, self.qref, self.doff, self.coff, self.module_index, self.module_pre_index]
        qoff = run_jet_core(args)

        """
        # Upper component's performance
        P00 = self.doff[5]
        W80 = qoff[1, self.module_pre_index]
        A80 = qoff[2, self.module_pre_index]
        TT80 = qoff[4, self.module_pre_index]
        PT80 = qoff[5, self.module_pre_index]
        tau80 = qoff[7, self.module_pre_index]
        pai80 = qoff[8, self.module_pre_index]

        gamma, cp = self.coff[2, self.module_index], self.coff[1, self.module_index]
        # calculate coefficients
        gamma_t, gamma_p, gamma_w, rg = calc_coef(cp, gamma)

        # Current component's performances
        W90 = W80
        TT90 = TT80 * tau80
        PT90 = PT80 * pai80

        # temperature and pressure ratio
        tau90 = 1.0
        pai90 = 1.0
        L90 = 0.0

        #########Judge Choke########
        pcr90 = PT90 / (1.0 + gamma_t) ** gamma_p

        if P00 <= pcr90:
            mc90 = 1.0
            P90 = pcr90

        else:
            mc90 = np.sqrt(((PT90 / P00) ** (1.0 / gamma_p) - 1.0) / gamma_t)
            P90 = P00
            # if np.isnan(mc90):
            #     mc90 = 1.0
            #     P90 = pcr90

        MFP90 = np.sqrt(gamma / rg) * mc90 / (1.0 + gamma_t * mc90 ** 2) ** gamma_w
        A90 = W90 / (PT90 / np.sqrt(TT90) * MFP90)
        U90 = mc90 * np.sqrt(gamma * rg * TT90 / (1.0 + gamma_t * mc90 ** 2))

        # results
        qoff[1, self.module_index:self.module_index + 1] = W90
        qoff[2, self.module_index:self.module_index + 1] = A90
        qoff[3, self.module_index:self.module_index + 1] = 1.0
        qoff[4, self.module_index:self.module_index + 1] = TT90
        qoff[5, self.module_index:self.module_index + 1] = PT90
        qoff[6, self.module_index:self.module_index + 1] = L90
        qoff[7, self.module_index:self.module_index + 1] = tau90
        qoff[8, self.module_index:self.module_index + 1] = pai90
        qoff[9, self.module_index:self.module_index + 1] = MFP90

        # Thrust
        qoff[0, self.module_index:self.module_index + 1] = W90 * U90 + A90 * (P90 - P00)
        """

        return qoff


class FanNozzle(InitEngineParamsCore):

    def __init__(self, coff, doff, qref, propulsion_type):
        super().__init__()

        self.coff = coff
        self.doff = doff
        self.qref = qref

        self.propulsion_type = propulsion_type

        # current module station number
        self.module_index = 18
        # previous curent module station number
        self.module_pre_index = 10

    def __call__(self, qoff, qoff_e, rev_rate, rev_lp):
        """
        args:qoff(all results array)
             rev_rate:fixied airflow rate
             rev_lp:revolve lp shaft(ex.0.95)
        """

        args = [qoff, self.qref, self.doff, self.coff, self.module_index, self.module_pre_index]
        qoff = run_fannozzle_core(args)

        """
        # Upper components performances
        W10 = qoff[1, self.module_pre_index]
        A10 = qoff[2, self.module_pre_index]
        TT10 = qoff[4, self.module_pre_index]
        PT10 = qoff[5, self.module_pre_index]
        tau10 = qoff[7, self.module_pre_index]
        pai10 = qoff[8, self.module_pre_index]

        # design point performance
        A18ref = self.qref[2, self.module_index]

        gamma, cp = self.coff[2, self.module_index], self.coff[1, self.module_index]
        gamma_t, gamma_p, gamma_w, rg = calc_coef(cp, gamma)

        TT18 = TT10 * tau10
        PT18 = PT10 * pai10
        A18 = A18ref

        # temperature ratio and pressure ratio
        tau18 = 1.0
        pai18 = self.coff[3, self.module_index]
        L18 = 0.0

        # results
        qoff[2, self.module_index:self.module_index + 1] = A18
        qoff[3, self.module_index:self.module_index + 1] = 0.0
        qoff[4, self.module_index:self.module_index + 1] = TT18
        qoff[5, self.module_index:self.module_index + 1] = PT18
        qoff[6, self.module_index:self.module_index + 1] = L18
        qoff[7, self.module_index:self.module_index + 1] = tau18
        qoff[8, self.module_index:self.module_index + 1] = pai18
        qoff[0, self.module_index:self.module_index + 1] = 0.0
        """

        return qoff, rev_rate, rev_lp


class FanJet(InitEngineParamsCore):

    def __init__(self, coff, doff, qref, propulsion_type):

        super().__init__()

        self.coff = coff
        self.doff = doff
        self.qref = qref

        self.propulsion_type = propulsion_type

        # current module station number
        self.module_index = 19
        # previous curent module station number
        self.module_pre_index = 18

    def __call__(self, qoff, qoff_e, rev_rate, rev_lp):

        """
        args:qoff(all results array)
             rev_rate:fixied airflow rate
             rev_lp:revolve lp shaft(ex.0.95)
        """

        args = [qoff, self.qref, self.doff, self.coff, self.module_index, self.module_pre_index]
        qoff = run_fanjet_core(args)

        """
        # Upper components
        P00 = self.doff[5]
        A18 = qoff[2, self.module_pre_index]
        TT18 = qoff[4, self.module_pre_index]
        PT18 = qoff[5, self.module_pre_index]
        tau18 = qoff[7, self.module_pre_index]
        pai18 = qoff[8, self.module_pre_index]

        # Design point performance
        A19ref = self.qref[2, self.module_index]

        gamma, cp = self.coff[2, self.module_index], self.coff[1, self.module_index]
        gamma_t, gamma_p, gamma_w, rg = calc_coef(cp, gamma)

        TT19 = TT18 * tau18
        PT19 = PT18 * pai18

        # temperature ratio, pressure ratio and entarpy
        tau19 = 1.0
        pai19 = 1.0
        L19 = 0.0

        ###########Judge choke######
        pcr19 = PT19 / (1.0 + gamma_t) ** gamma_p

        # Equal to design point
        A19off = A19ref * self.doff[45]

        if P00 <= pcr19:

            mc19 = 1.0
            P19 = pcr19

        else:

            mc19 = np.sqrt(((PT19 / P00) ** (1.0 / gamma_p) - 1.0) / gamma_t)
            P19 = P00

        MFP19 = np.sqrt(gamma / rg) * mc19 / (1.0 + gamma_t * mc19 ** 2) ** gamma_w
        A19 = A19off
        W19 = PT19 * A19 / np.sqrt(TT19) * MFP19
        U19 = mc19 * np.sqrt(gamma * rg * TT19 / (1.0 + gamma_t * mc19 ** 2))

        # results
        # FanNozzle
        W18 = W19
        MFP18 = W18 * np.sqrt(TT18) / (PT18 * A18)
        qoff[1, self.module_pre_index:self.module_pre_index + 1] = W18
        qoff[9, self.module_pre_index:self.module_pre_index + 1] = MFP18

        # FanJet
        qoff[1, self.module_index:self.module_index + 1] = W19
        qoff[2, self.module_index:self.module_index + 1] = A19
        qoff[3, self.module_index:self.module_index + 1] = 0.0
        qoff[4, self.module_index:self.module_index + 1] = TT19
        qoff[5, self.module_index:self.module_index + 1] = PT19
        qoff[6, self.module_index:self.module_index + 1] = L19
        qoff[7, self.module_index:self.module_index + 1] = tau19
        qoff[8, self.module_index:self.module_index + 1] = pai19
        qoff[9, self.module_index:self.module_index + 1] = MFP19

        # Thrust
        F19 = W19 * U19 + A19 * (P19 - P00)

        qoff[0, self.module_index:self.module_index + 1] = F19
        """

        return qoff, rev_rate, rev_lp


##################################Electric Part####################################

class InletElec(InitEngineParamsCore):

    def __init__(self, coff, doff, coff_e, qref, qref_e, propulsion_type):
        super().__init__()

        self.propulsion_type = propulsion_type

        self.coff = coff
        self.doff = doff
        self.coff_e = coff_e
        self.qref = qref
        self.qref_e = qref_e

        # current module station number
        self.module_index = 0
        # previous curent module station number
        self.module_pre_index = None

    def __call__(self, qoff_e, rev_rate, rev_fan):
        """
        args:qoff(all results array)
        """

        args = [qoff_e, self.doff, self.coff, self.module_index]
        qoff_e = run_inlet_elec(args)

        """
        gamma, cp = self.coff[2, self.module_index], self.coff[1, self.module_index]
        gamma_t, gamma_p, gamma_w, rg = calc_coef(cp, gamma)

        mc00 = self.doff[1]
        T00 = self.doff[4]
        P00 = self.doff[5]

        TT00 = T00 * (1.0 + gamma_t * mc00 ** 2)
        PT00 = P00 * (1.0 + gamma_t * mc00 ** 2) ** gamma_p
        MFP00 = np.sqrt(gamma / rg) * mc00 / (1.0 + gamma_t * mc00 ** 2) ** gamma_w
        U00 = mc00 * np.sqrt(gamma * rg * TT00 / (1.0 + gamma_t * mc00 ** 2))

        ytd00 = self.coff[3, self.module_index]
        pai00 = 1.0 / ((1.0 + (1.0 - ytd00) * gamma_t * mc00 ** 2) ** gamma_p)
        tau00 = 1.0

        # results
        qoff_e[3, self.module_index] = 0.0
        qoff_e[4, self.module_index] = TT00
        qoff_e[5, self.module_index] = PT00
        qoff_e[6, self.module_index] = 0.0
        qoff_e[7, self.module_index] = tau00
        qoff_e[8, self.module_index] = pai00
        qoff_e[9, self.module_index] = MFP00
        qoff_e[10, self.module_index] = U00
        """

        return qoff_e, rev_rate, rev_fan


class FanElec(InitEngineParamsCore):

    def __init__(self, coff, doff, coff_e, qref, qref_e, propulsion_type):
        super().__init__()

        self.coff = coff
        self.doff = doff
        self.coff_e = coff_e
        self.qref = qref
        self.qref_e = qref_e

        self.propulsion_type = propulsion_type

        # current module station number
        self.module_index = 10
        # previous curent module station number
        self.module_pre_index = 0

    def __call__(self, qoff_e, rev_rate, rev_fan):
        """
        args:qoff(all results array)
            rev_rate:angle(rad)
        """
        args = [qoff_e, self.qref_e, self.doff, self.coff, self.coff_e, rev_rate, self.module_index, self.module_pre_index]
        qoff_e = run_fan_elec(args)

        """
        # Upper component's performances
        U00 = qoff_e[10, self.module_pre_index]
        TT00 = qoff_e[4, self.module_pre_index]
        PT00 = qoff_e[5, self.module_pre_index]
        tau00 = qoff_e[7, self.module_pre_index]
        pai00 = qoff_e[8, self.module_pre_index]
        MFP00 = qoff_e[9, self.module_pre_index]

        # Design point performances
        A10 = self.qref_e[2, self.module_index]
        ytp10ref = self.coff_e[3, self.module_index]
        tau10ref = self.qref_e[7, self.module_index]
        pai10ref = self.qref_e[8, self.module_index]
        MFP10ref = self.qref_e[9, self.module_index]

        gamma, cp = self.coff[2, self.module_index], self.coff[1, self.module_index]
        gamma_t, gamma_p, gamma_w, rg = calc_coef(cp, gamma)

        # Current component's performances
        TT10 = TT00 * tau00
        PT10 = PT00 * pai00
        prmlp10 = rev_rate
        prmwl10 = rev_rate

        alpha, beta = self.coff_e[4, self.module_index], self.coff_e[5, self.module_index]

        # Calculate operating point on the compressor map
        prmt10, pai10, ytc10 = calc_tau_pai_ratio(prmwl10, prmlp10, alpha, beta, gamma_p, ytp10ref, pai10ref, tau10ref)

        # Temperature and Pressure ratio
        MFP10 = MFP10ref * prmwl10
        W10 = MFP10 * (PT10 * A10) / np.sqrt(TT10)
        tau10 = (tau10ref - 1.0) * prmt10 + 1.0
        L10 = W10 * cp * TT10 * (tau10ref - 1.0) * prmt10

        # results
        W00 = W10
        A00 = W00 / (PT00 / np.sqrt(TT00) * max(MFP00, 1.0e-10))

        # Inlet
        qoff_e[1, self.module_pre_index] = W00
        qoff_e[2, self.module_pre_index] = A00
        qoff_e[0, self.module_pre_index] = -W00 * U00

        # Fan
        qoff_e[1, self.module_index] = W10
        qoff_e[2, self.module_index] = A10
        qoff_e[3, self.module_index] = prmlp10
        qoff_e[4, self.module_index] = TT10
        qoff_e[5, self.module_index] = PT10
        qoff_e[6, self.module_index] = L10
        qoff_e[7, self.module_index] = tau10
        qoff_e[8, self.module_index] = pai10
        qoff_e[9, self.module_index] = MFP10

        qoff_e[0, self.module_index] = ytc10
        """

        return qoff_e, rev_rate, rev_fan


class FanNozzleElec(InitEngineParamsCore):

    def __init__(self, coff, doff, coff_e, qref, qref_e, propulsion_type):
        super().__init__()

        self.coff = coff
        self.doff = doff
        self.coff_e = coff_e
        self.qref = qref
        self.qref_e = qref_e

        self.propulsion_type = propulsion_type

        # current module station number
        self.module_index = 18
        # previous curent module station number
        self.module_pre_index = 10

    def __call__(self, qoff_e, rev_rate, rev_fan):
        """
        args:qoff(all results array)
             rev_rate:fixied airflow rate
             rev_fan:revolve rate (ex.0.95)
        """

        args = [qoff_e, self.qref_e, self.coff, self.module_index, self.module_pre_index]
        qoff_e = run_fannozzle_elec(args)

        """
        # Upper Component's performances
        W10 = qoff_e[1, self.module_pre_index]
        A10 = qoff_e[2, self.module_pre_index]
        TT10 = qoff_e[4, self.module_pre_index]
        PT10 = qoff_e[5, self.module_pre_index]
        tau10 = qoff_e[7, self.module_pre_index]
        pai10 = qoff_e[8, self.module_pre_index]

        # Design point performances
        A18ref = self.qref_e[2, self.module_index]

        gamma, cp = self.coff[2, self.module_index], self.coff[1, self.module_index]
        gamma_t, gamma_p, gamma_w, rg = calc_coef(cp, gamma)

        # Current component's performances
        W18 = W10
        TT18 = TT10 * tau10
        PT18 = PT10 * pai10
        A18 = A18ref
        MFP18 = W18 * np.sqrt(TT18) / (PT18 * A18)

        # temperature and pressure ratio
        tau18 = 1.0
        pai18 = self.coff[3, self.module_index]
        L18 = 0.0

        # results
        qoff_e[1, self.module_index] = W18
        qoff_e[2, self.module_index] = A18
        qoff_e[3, self.module_index] = 0.0
        qoff_e[4, self.module_index] = TT18
        qoff_e[5, self.module_index] = PT18
        qoff_e[6, self.module_index] = L18
        qoff_e[7, self.module_index] = tau18
        qoff_e[8, self.module_index] = pai18
        qoff_e[9, self.module_index] = MFP18

        qoff_e[0, self.module_index] = 0.0
        """

        return qoff_e, rev_rate, rev_fan


class FanJetElec(InitEngineParamsCore):

    def __init__(self, coff, doff, coff_e, qref, qref_e, propulsion_type):

        super().__init__()

        self.coff = coff
        self.doff = doff
        self.coff_e = coff_e
        self.qref = qref
        self.qref_e = qref_e

        self.propulsion_type = propulsion_type

        # current module station number
        self.module_index = 19
        # previous curent module station number
        self.module_pre_index = 18

    def __call__(self, qoff_e, rev_rate, rev_fan):

        """
        args:qoff(all results array)
             rev_rate:fixied airflow rate
             rev_fan:revolve fan(ex.0.95)
        """

        args = [qoff_e, self.qref_e, self.doff, self.coff, self.module_index, self.module_pre_index]
        qoff_e = run_fanjet_elec(args)

        """
        # Upper Component's performances
        P00 = self.doff[5]
        W18 = qoff_e[1, self.module_pre_index]
        A18 = qoff_e[2, self.module_pre_index]
        TT18 = qoff_e[4, self.module_pre_index]
        PT18 = qoff_e[5, self.module_pre_index]
        tau18 = qoff_e[7, self.module_pre_index]
        pai18 = qoff_e[8, self.module_pre_index]

        gamma, cp = self.coff[2, self.module_index], self.coff[1, self.module_index]
        gamma_t, gamma_p, gamma_w, rg = calc_coef(cp, gamma)

        # Current Component's performances
        W19 = W18
        TT19 = TT18 * tau18
        PT19 = PT18 * pai18

        # temperature and pressure ratio
        tau19 = 1.0
        pai19 = 1.0
        L19 = 0.0

        #########Judge Choke#########
        pcr19 = PT19 / (1.0 + gamma_t) ** gamma_p

        if P00 <= pcr19:

            mc19 = 1.0
            P19 = pcr19

        else:
            # mc19 = np.sqrt(((PT19 / P00) ** (1.0 / gamma_p) - 1.0) / gamma_t)
            # P19 = P00
            mc19 = 1.0
            P19 = P00

        MFP19 = np.sqrt(gamma / rg) * mc19 / (1.0 + gamma_t * mc19 ** 2) ** gamma_w
        A19 = W19 / (PT19 / np.sqrt(TT19) * MFP19)
        U19 = mc19 * np.sqrt(gamma * rg * TT19 / (1.0 + gamma_t * mc19 ** 2))

        # results
        qoff_e[1, self.module_index] = W19
        qoff_e[2, self.module_index] = A19
        qoff_e[3, self.module_index] = 1.0
        qoff_e[4, self.module_index] = TT19
        qoff_e[5, self.module_index] = PT19
        qoff_e[6, self.module_index] = L19
        qoff_e[7, self.module_index] = tau19
        qoff_e[8, self.module_index] = pai19
        qoff_e[9, self.module_index] = MFP19

        # Thrust
        qoff_e[0, self.module_index] = W19 * U19 + A19 * (P19 - P00)
        """

        return qoff_e, rev_rate, rev_fan
