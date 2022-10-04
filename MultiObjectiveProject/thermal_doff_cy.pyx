import numpy as np
cimport numpy as np
import time
cimport cython
from EngineComponentOff import *
# from off_design_integ_utils import core_off_compute, electric_off_compute, core_lp_former_compute, core_hp_compute, core_lp_latter_compute
ctypedef np.float64_t DOUBLE_t

# Helper function for class Ca;cDesignOffPoint
# Divide LP HP components
cpdef get_component_classes(str propulsion_type):
    """
    data structure of component class list

    Attributes
    ------------
    propulsion_type: str
                     the name of propulsion system
    lpshaft: list
            low pressure part components

            ex) [[former components],[latter components]]

    hpshaft: list
            high pressure part components

            ex) [current components]

    elecshaft: list
            distributed fan part components

            ex)[current components]

    """

    lpshaft, hpshaft, elecshaft = None, None, None

    # If you research another propulsion type, you have to divide components into two parts
    # You should consider how to match energy consumption of each shapfts(LP,HP,Electric)
    # You had better draw overview of components of propulsion systems. Example I will post at ReadMe.md
    if propulsion_type == 'turbojet' or propulsion_type == 'turboshaft':

        lpshaft = [[Inlet, LPC, HPC], [HPTCool, LPT, LPTCool, CoreOut, Nozzle, Jet]]
        hpshaft = [HPC, CC, HPTCool]

    elif propulsion_type == 'turbofan':

        lpshaft = [[Inlet, Fan, FanNozzle, FanJet, LPC, HPC], [HPTCool, LPT, LPTCool, CoreOut, Nozzle, Jet]]
        hpshaft = [HPC, CC, HPT]

    elif propulsion_type == 'TeDP':

        lpshaft = [[Inlet, LPC, HPC], [HPTCool, LPT, LPTCool, CoreOut, Nozzle, Jet]]
        hpshaft = [HPC, CC, HPT]
        elecshaft = [InletElec, FanElec, FanNozzleElec, FanJetElec]

    elif propulsion_type == 'PartialElectric':

        lpshaft = [[Inlet, Fan, FanNozzle, FanJet, LPC, HPC], [HPTCool, LPT, LPTCool, CoreOut, Nozzle, Jet]]
        hpshaft = [HPC, CC, HPT]
        elecshaft = [InletElec, FanElec, FanNozzleElec, FanJetElec]

    return lpshaft, hpshaft, elecshaft


class CalcDesignOffPoint(object):
    """
    Note:
        Computational Algorithm

        1. set the values of engine efficiency and design variables at the design point
        2. assume the ratio of revolving rate at the low pressure side against that at the design point (1.0 at the design point)
        3. calculate the performances of former part of low pressure side by the same algorithm at the design point
        4. apart from 3. process, fixed flow rate, pressure ratio and temperature ratio are needed.
        5. in the premise that compressor curve fits the eclipse curve, indexes of 4. will have been calculated
        6. after finishing the computation of low pressure side, assume the ratio of revolving rate at the high pressure side against that at the design point
        7. calculate the performances of high pressure side by the same algorithm at the design point
        8. fixed flow rate, pressure ratio and temperature ratio compute from the compressor map
        9. by the entropy of high pressure compressor and turbine, conduct the energy matching and determine the revolving ratio of high pressure
        10. calculate the performances of latter part of low pressure side
        11. by the areas of jet nozzle both at the design point and the off design point, the convergence of low pressure side needs to be checked
        12. if 11. process succeeds, be able to determine the revolving ratio of low pressure side and finished this computations


    Attributes
    ----------------
    calc_design_point_class: class object
                             the class object for calculating and restoring results of thermal performances
                             at the design point
    aircraft_name: str
                   aircraft name under investigation

                   ex) 'A320'
    engine_name: str
                 engine name under investigation

                 ex) 'V2500'
    aircraft_type: str
                   aircraft type under investigation

                   ex) 'normal'
    propulsion_type: str
                     type of propulsion system under investigation

                     ex) 'turbofan'
    data_base_args: list
                    the list of data base paths => [aircraft, engine, missions]

    cref: numpy array
          the array which has data such as element efficiency values on the engine side at the design point

          its dimension is 2d

    cref_e: numpy array
            the array which has data such as element efficiency values on the distributed side at the design point

            its dimension is 2d

    dref: numpy array
          the array which has data such as design variables at the design point

          its dimension is 1d

    qref: numpy array
          the array of the collection of results at the each component in the core side of engine

          its dimension is 2d

    qref_e: numpy array
            the array of putting the results of each engine component in the distributed side toghther

            its dimension is 2d

    fscl: float
          airflow ratio at the design point
    sfc: float
         specific fuel consumption, in short, the efficiency of usage of fuel

    isp: float
         specific thrust, in short, the ratio of thrust against fuel consumption
    airflow: float
             the volume of air through the core engine
    A_core: float
            the cross sectional area of core engine
    core_diam: float
               the diameter of front surface at the core engine side
    disfan_diam: float
               the diameter of front surface at the distributed engine side

    coff: numpy array
          the array of the values of core engine efficiency at the off design point

          basically, this array is equal to the 'cref'
    coff_e: numpy array
            the array of the values of distributed engine efficiency at the off design point

            basically, this array is same as the 'cref_e'

    doff: numpy array
          the array of the values of design variables at the off design point

          but, this array is a little bit different. If you want to acquire them,

          you will have to assume the ratio of revolving at the off design point against it at the design point

          and conduct energy-matching calculation on both low pressure side and high one

    lp_shaft_classes: list
                      the names of low pressure components
    hp_shaft_classes: list
                      the names of high pressure components
    elec_shaft_classes: list
                      the names of electric side components, in most cases, electric side components

                      have only low pressure side's features

    """

    def __init__(self, aircraft_name, engine_name, aircraft_type, propulsion_type, thermal_design_variables,
                 off_param_args, design_point_params, data_base_args, calc_design_point_class):

        self.design_point_params = design_point_params
        self.aircraft_data_path, self.engine_data_path, self.mission_data_path = data_base_args
        # define design point class
        self.calc_design_point_class = calc_design_point_class
        # define other arguments
        self.aircraft_name = aircraft_name
        self.engine_name = engine_name
        self.aircraft_type = aircraft_type
        self.propulsion_type = propulsion_type
        self.data_base_args = data_base_args
        # Already built class
        self.cref = self.calc_design_point_class.cref
        self.cref_e = self.calc_design_point_class.cref_e
        self.dref = self.calc_design_point_class.dref
        self.qref = self.calc_design_point_class.qref
        self.qref_e = self.calc_design_point_class.qref_e

        # objective index
        self.fscl = self.calc_design_point_class.fscl
        self.sfc = self.calc_design_point_class.sfc
        self.isp = self.calc_design_point_class.isp
        self.airflow = self.calc_design_point_class.airflow
        self.A_core = self.calc_design_point_class.A_core
        self.core_diam = self.calc_design_point_class.core_diam
        self.disfan_diam = self.calc_design_point_class.disfan_diam

        # set the design off point params
        self.calc_design_point_class.define_design_off_point(off_param_args)

        self.coff = self.calc_design_point_class.coff
        self.coff_e = self.calc_design_point_class.coff_e
        self.doff = self.calc_design_point_class.doff

        # get LP,HP,Electric shaft class objects
        self.lp_shaft_classes, self.hp_shaft_classes, self.elec_shaft_classes = get_component_classes(propulsion_type)

        # Gravity
        self.g = 9.81

        # Set compressor map params
        self.set_params_for_op_at_cmap()

        # Build class objects
        self.build_off_design_component_classes()

    def set_params_for_op_at_cmap(self):
        """
        Find Operating point on th compressor map by changing angle
        """
        ######## Angles ########

        # Initial angle lp
        self.init_angle_lp = 45.0 * np.pi / 180.0
        # Initial angle lp step
        self.init_angle_lp_step = -1.0 * np.pi / 180.0
        # Initial angle hp shaft
        self.init_angle_hp = 40.0 * np.pi / 180.0
        # Initial angle hp shaft step
        self.init_angle_hp_step = -1.0 * np.pi / 180.0

        # Initial electric angle
        self.init_angle_elec = 40.0 * np.pi / 180.0
        # Initial electric angle step
        self.init_angle_elec_step = -0.1 * np.pi / 180.0

        # Initial distributed LP HP shaft ratio step
        self.div_alpha_step = 0.03

        ##### Matching Target Epsilon ####

        # LPshaft convergence target epsilon
        self.epslp = 1.0e-7
        # HPshaft convergence target epsilon
        self.epshp = 1.0e-7  # comparison to turbofan, loose the limit  : 1.0e-8
        # Electric shaft convergence energy target epsilon
        self.epsele = 1.0e-7
        # Electric shaft convergence fanarea target epsilon
        self.eps_fanarea = 1.0e-4  # comparison to turbofan, loose the limit  : 1.0e-5

    def build_off_design_component_classes(self):
        """
        construct component classes and integrates into the total calculation class

        :return:
        """

        args = [self.lp_shaft_classes, self.hp_shaft_classes, self.elec_shaft_classes, self.coff, self.doff, self.qref, self.coff_e, self.qref_e, self.propulsion_type]
        self.build_hp_classes, self.build_lp_former_classes, self.build_lp_latter_classes, self.build_electric_classes = run_build_classes(args)


    # calculate off design point
    def run_off(self, rev_args):
        """

        :param rev_args: list
               [revolving rate at the core side, revolving rate at the distributed side]
        :return: None
        """

        # print('run off design ')
        args = [rev_args, self.propulsion_type, self.run_core_off, self.run_electric_off, self.run_battery_off]
        self.qoff, self.qoff_e = main_core(args)


    def run_core_off(self, qoff_e, rev_lp):
        """
        while lp shaft enegy consumptions are matched, repeat run_lp function by changing angle_lp on the compressor map
        Watch out run_lp function includes ryn_hp function

        :param rev_lp (ratio of revolving lp shaft against the design point's revolving params as 1.0)
        :return: qoff
        """

        qoff = np.zeros((100, 100))

        self.rev_lp = rev_lp
        args = [qoff, qoff_e, self.rev_lp, self.qref, self.coff, self.doff, self.init_angle_lp, self.init_angle_lp_step,
                self.init_angle_hp, self.init_angle_hp_step, self.epslp, self.epshp, self.propulsion_type, self.run_lp_former, self.run_hp, self.run_lp_latter]
        qoff = core_off_compute(args)

        return qoff

    # Before converging HP shaft
    def run_lp_former(self, qoff, qoff_e, rev_rate, rev_lp):
        """
        calculate performances at the former part of low pressure components
        :param qoff:
        :param rev_rate:
        :param rev_lp:
        :return: qoff, rev_rate, rev_lp
        """
        args = [qoff, qoff_e, rev_rate, rev_lp, self.build_lp_former_classes]
        qoff, rev_rate, rev_lp = core_lp_former_compute(args)

        return qoff, rev_rate, rev_lp

    # After converging HP shaft
    def run_lp_latter(self, qoff, qoff_e):
        """
        calculate performances at the latter part of low pressure components

        :param qoff:
        :return: qoff
        """
        args = [qoff, qoff_e, self.build_lp_latter_classes]
        qoff = core_lp_latter_compute(args)

        return qoff

    def run_hp(self, qoff, qoff_e, rev_rate, rev_hp):
        """
        while hp shaft energy consumptions are matched, repeat run_hp function by changing angle_hp on the compressor map

        :param qoff:
        :param rev_rate:
        :param rev_hp:

        :return: qoff, rev_rate, rev_hp

        """
        args = [qoff, qoff_e, rev_rate, rev_hp, self.build_hp_classes]
        qoff, rev_rate, rev_hp = core_hp_compute(args)

        return qoff, rev_rate, rev_hp

    # To Do: Now implementd except for propulsion system equipping with batteries
    def run_electric_off(self, rev_args):
        """
        Matching for distribution ratio of core and subsystem

        :param rev_args: list of the values of current revolving rate
        :return: qoff, qoff_e
        """

        args = [self.doff, self.div_alpha_step, rev_args, self.qref_e, self.eps_fanarea, self.run_distributed_fan, self.run_core_off, self.build_off_design_component_classes]
        qoff, qoff_e, self.doff = electric_off_compute(args)

        return qoff, qoff_e


    def run_distributed_fan(self, rev_fan):
        """
        energy matching at the part of distributed fan (subsystem)

        :param qoff:
        :param rev_fan:
        :return: qoff_e
        """


        qoff_e = np.zeros((100, 100))

        rev_ele_rate = rev_fan

        for e_class in self.build_electric_classes:
            qoff_e, rev_ele_rate, rev_fan = e_class(qoff_e, rev_ele_rate, rev_fan)

        return qoff_e

    # Engine with batteries
    def run_battery_off(self):
        """
        energy matching for battery part

        :return: NotImplementedError
        """

        raise NotImplementedError()

    # objective_func
    def objective_func_doff(self):
        """
        calculate the values of objective indexes and other variables

        :return: None
        """

        args = [self.qoff, self.qoff_e, self.qref, self.qref_e, self.doff, self.fscl, self.g, self.propulsion_type]

        self.GenePower, self.COT, self.TIT, self.sfc_off, self.isp_off, self.fpr_off, self.thrust_off, self.airflow_off = objective_func(args)



cpdef calc_off_design_point(str aircraft_name, str engine_name, str aircraft_type, str propulsion_type, list thermal_design_variables,
                              list off_param_args, list design_point_params, list data_base_args, calc_design_point_class, list rev_args):
    # define design off point class
    cdop = CalcDesignOffPoint(aircraft_name, engine_name, aircraft_type, propulsion_type, thermal_design_variables,
                              off_param_args, design_point_params, data_base_args, calc_design_point_class)
    # calculating off design point
    cdop.run_off(rev_args)
    # calculating important indexes
    cdop.objective_func_doff()

    return cdop

### outside function####
cdef main_core(list args):
    cdef:
        np.ndarray[DOUBLE_t, ndim=2] qoff
        np.ndarray[DOUBLE_t, ndim=2] qoff_e
        list rev_args
        str propulsion_type
        double rev_lp
        double rev_fan

    rev_args, propulsion_type, run_core_off, run_electric_off, run_battery_off = args
    qoff, qoff_e = np.zeros((100, 100)), np.zeros((100, 100))
    rev_lp, rev_fan = rev_args
    # Only Jet Engine
    if propulsion_type in ['turbojet', 'turboshaft', 'turbofan']:
        qoff = run_core_off(qoff_e, rev_lp)

    # Jet Engine + Electric
    if propulsion_type in ['TeDP', 'PartialElectric']:
        qoff, qoff_e = run_electric_off(rev_args)

    # Jet Engine + Battery
    if propulsion_type in ['battery', 'hybridturbojet', 'hybridturbofan']:
        qoff, qoff_e = run_battery_off()

    return qoff, qoff_e


cdef core_lp_former_compute(list args):
    cdef np.ndarray[DOUBLE_t, ndim=2] qoff
    cdef np.ndarray[DOUBLE_t, ndim=2] qoff_e
    cdef double rev_rate
    cdef double rev_lp
    cdef list build_lp_former_classes
    cdef int len_lf
    cdef int idx

    qoff, qoff_e, rev_rate, rev_lp, build_lp_former_classes = args
    len_lf = len(build_lp_former_classes)

    for idx in range(len_lf):
        lp_f_class = build_lp_former_classes[idx]
        qoff, rev_rate, rev_lp = lp_f_class(qoff, qoff_e, rev_rate, rev_lp)

    return qoff, rev_rate, rev_lp

cdef core_hp_compute(list args):
    cdef np.ndarray[DOUBLE_t, ndim=2] qoff
    cdef np.ndarray[DOUBLE_t, ndim=2] qoff_e
    cdef double rev_rate
    cdef double rev_hp
    cdef list build_hp_classes
    cdef int len_hp
    cdef int idx

    qoff, qoff_e, rev_rate, rev_hp, build_hp_classes = args
    len_hp = len(build_hp_classes)
    for idx in range(len_hp):
        hp_class = build_hp_classes[idx]
        qoff, rev_rate, rev_hp = hp_class(qoff, qoff_e, rev_rate, rev_hp)

    return qoff, rev_rate, rev_hp

cdef core_lp_latter_compute(list args):
    cdef np.ndarray[DOUBLE_t, ndim=2] qoff
    cdef np.ndarray[DOUBLE_t, ndim=2] qoff_e
    cdef double rev_rate
    cdef double rev_lp
    cdef list build_lp_latter_classes
    cdef int len_lf
    cdef int idx

    qoff, qoff_e, build_lp_latter_classes = args
    len_lf = len(build_lp_latter_classes)

    for idx in range(len_lf):
        lp_f_class = build_lp_latter_classes[idx]
        qoff = lp_f_class(qoff, qoff_e)

    return qoff


cdef core_off_compute(list args):
    cdef np.ndarray[DOUBLE_t, ndim=2] qoff
    cdef np.ndarray[DOUBLE_t, ndim=2] qoff_e
    cdef np.ndarray[DOUBLE_t, ndim=2] qref
    cdef np.ndarray[DOUBLE_t, ndim=2] coff
    cdef np.ndarray[DOUBLE_t, ndim=1] doff
    cdef double base_rev_lp
    cdef double init_angle_lp
    cdef double init_angle_lp_step
    cdef double init_angle_hp
    cdef double init_angle_hp_step
    cdef double eps_lp
    cdef double eps_hp
    cdef double beta
    cdef double reslp
    cdef double reslpold
    cdef int iterlp
    cdef double rev_lp_rate
    cdef double reshp
    cdef double reshpold
    cdef int iterhp
    cdef double angle_lp
    cdef double angle_lp_step
    cdef double angle_hp
    cdef double angle_hp_step
    cdef double rev_rate
    cdef double rev_rate25
    cdef double rev_hp
    cdef double ytm40
    cdef double ytdah
    cdef double ytd25
    cdef double L25
    cdef double L40
    cdef double A90
    cdef double A90_target

    cdef str propulsion_type

    # set arguments
    qoff, qoff_e, base_rev_lp, qref, coff, doff, init_angle_lp, init_angle_lp_step, init_angle_hp, \
    init_angle_hp_step, eps_lp, eps_hp, propulsion_type, run_lp_former, run_hp, run_lp_latter = args

    if propulsion_type in ['turbojet', 'turboshaft', 'TeDP']:
        beta = coff[7, 20]
    elif propulsion_type in ['turbofan', 'PartialElectric']:
        beta = coff[5, 10]
    A90_target = qref[2, 90]

    angle_lp = init_angle_lp
    angle_lp_step = init_angle_lp_step

    # residual lp shaft
    # Convergence by applying two split method
    reslp = 0.0
    reslpold = 0.0
    iterlp = 0

    lp_ok = True

    while lp_ok:
        # prmwl == rev_lp_rate
        rev_lp_rate = base_rev_lp * (np.sqrt(2.0) * np.cos(angle_lp)) ** (2.0 / beta)

        iterlp += 1

        # run_former()
        qoff, rev_rate, rev_lp = run_lp_former(qoff, qoff_e, rev_lp_rate, base_rev_lp)

        if propulsion_type in ['turbojet', 'turboshaft', 'TeDP']:
            beta = coff[7, 25]
        elif propulsion_type in ['turbofan', 'PartialElectric']:
            beta = coff[5, 25]

        # residual hp shaft
        # Convergence by applying two split method
        reshp = 0.0
        reshpold = 0.0
        iterhp = 0
        # Initializr angle hp shaft
        angle_hp = init_angle_hp
        angle_hp_step = init_angle_hp_step
        # print('angle_hp:',angle_hp)
        rev_rate25 = rev_rate
        rev_hp = rev_rate25 / (np.sqrt(2.0) * np.cos(angle_hp)) ** (2.0 / beta)  # HPC rev_rate
        hp_ok = True

        while hp_ok:

            iterhp += 1

            # Running Hp shaft components class objects
            qoff, rev_rate, rev_hp = run_hp(qoff, qoff_e, rev_rate25, rev_hp)

            # prepare for checking energy matching
            ytm40 = coff[5, 40]
            ytdah = coff[4, 25]
            ytd25 = doff[51]

            # Entarpy
            L25 = qoff[6, 25]
            L40 = qoff[6, 40]

            # Residual entarpy
            reshp = (L40 + (L25) / (ytm40 + ytd25)) / L40

            # confirmation for HP shaft energy matching
            # print('='*10 + 'HP shaft energy matching' + '='*10)
            # print('L40:',L40,'L25:',L25,'reshp:',reshp)

            # Error process
            if iterhp == 10000:
                print('Over Computation!!')
                break

            if np.isnan(reshp):
                print('Computation is diffused!!')
                break

            # Normal Convergence
            if abs(reshp) < eps_hp:
                hp_ok = False

            # step is diminishing if the current value is beyond the target value
            if reshp * reshpold <= 0.0:
                angle_hp_step *= 0.5

            # Redefine HP shaft revolve
            if propulsion_type in ['turbojet', 'turboshaft', 'TeDP']:
                beta = coff[7, 25]
            elif propulsion_type in ['turbofan', 'PartialElectric']:
                beta = coff[5, 25]

            # update angle hp shaft
            angle_hp += np.sign(reshp) * angle_hp_step

            # restore old residual hp shaft
            reshpold = reshp

            # Replace revolve hp shaft

            rev_hp = rev_rate25 / (np.sqrt(2.0) * np.cos(angle_hp)) ** (2.0 / beta)

        # After hp shaft convergence
        qoff = run_lp_latter(qoff, qoff_e)

        # LP convergence
        A90 = qoff[2, 90]

        # residual lp shaft
        reslp = 1.0 - A90 / A90_target

        # print('A90:',A90,'A90_target:',A90_target,'reslp:',reslp)

        # Error process
        if iterlp == 1000:
            print('Over computation')
            break

        if np.isnan(reslp):
            print('Computation is diffused')

            break

        # Normal Convergence
        if abs(reslp) < eps_lp:
            lp_ok = False

        # Step is diminishing around the nearest solutions
        if reslp * reslpold <= 0.0:
            angle_lp_step *= 0.5

        # update lp shaft angle
        angle_lp += -np.sign(reslp) * angle_lp_step


        # restore old lp residual
        reslpold = reslp

        if propulsion_type in ['turbojet', 'turboshaft', 'TeDP']:
            beta = coff[7, 20]
        elif propulsion_type in ['turbofan', 'PartialElectric']:
            beta = coff[5, 10]


    return qoff


cdef electric_off_compute(list args):
    cdef np.ndarray[DOUBLE_t, ndim=1] doff
    cdef double div_alpha_step
    cdef np.ndarray[DOUBLE_t, ndim=2] qref_e
    cdef double eos_fanarea
    cdef double div_alpha
    cdef double res_fanarea
    cdef double res_fanareaold
    cdef int count
    cdef double rev_fan
    cdef double rev_lp
    cdef np.ndarray[DOUBLE_t, ndim=2] qoff_e
    cdef np.ndarray[DOUBLE_t, ndim=2] qoff
    cdef double A19
    cdef double A19_target

    doff, div_alpha_step, rev_args, qref_e, eps_fanarea, run_distributed_fan, run_core_off, build_off_design_component_classes = args
    # LP HP distirbuted ratio
    div_alpha = doff[33]
    div_alpha_step = div_alpha_step

    # Converge two split method
    res_fanarea = 0
    res_fanareaold = 0
    count = 0

    rev_lp, rev_fan = rev_args

    while True:
        # fan ratio
        rev_fan = rev_lp * div_alpha

        qoff_e = run_distributed_fan(rev_fan)

        qoff = run_core_off(qoff_e, rev_lp)

        # Residual fan area
        A19 = qoff_e[2, 19]
        A19_target = qref_e[2, 19]

        res_fanarea = 1.0 - A19 / A19_target

        # check convergence
        # print('A19:', A19, 'A19_target:', A19_target, 'res_fanarea:', res_fanarea, 'div_alpha:', div_alpha)

        # Error process
        if np.isnan(res_fanarea):
            break

        if count >= 100:
            print(count)
            # initialize results array
            qoff = np.zeros((100, 100))
            qoff_e = np.zeros((100, 100))
            break

        # Normal convergence
        if abs(res_fanarea) < eps_fanarea:
            break

        # Diminishing step
        if res_fanarea * res_fanareaold <= 0.0:
            div_alpha_step *= 0.5

        # Update div_alpha
        div_alpha += np.sign(res_fanarea) * div_alpha_step
        doff[33] = div_alpha


        # Rebuild class objects
        # build_off_design_component_classes()

        # restore old residual
        res_fanareaold = res_fanarea

        count += 1


    return qoff, qoff_e, doff

cdef run_build_classes(list args):
    cdef:
        list lp_shaft_classes
        list hp_shaft_classes
        list elec_shaft_classes
        np.ndarray[DOUBLE_t, ndim=2] coff
        np.ndarray[DOUBLE_t, ndim=1] doff
        np.ndarray[DOUBLE_t, ndim=2] qref
        np.ndarray[DOUBLE_t, ndim=2] coff_e
        np.ndarray[DOUBLE_t, ndim=2] qref_e
        str propulsion_type
        list build_hp_classes
        list build_lp_former_classes
        list build_lp_latter_classes
        list build_electric_classes
        int idx

    lp_shaft_classes, hp_shaft_classes, elec_shaft_classes, coff, doff, qref, coff_e, qref_e, propulsion_type = args

    build_hp_classes = []
    build_lp_former_classes = []
    build_lp_latter_classes = []
    build_electric_classes = []

    if hp_shaft_classes != None:
        # Build Hp class

        for idx, hp_class in enumerate(hp_shaft_classes):
            # HPC class switch
            if idx == 0:

                c_class = hp_class(coff, doff, qref, propulsion_type, False)

            else:

                c_class = hp_class(coff, doff, qref, propulsion_type)

            build_hp_classes.append(c_class)

    if lp_shaft_classes != None:
        # Build off design components class objects
        lp_former_classes = lp_shaft_classes[0]
        lp_latter_classes = lp_shaft_classes[1]

        # Build lp former class

        for lp_f_class in lp_former_classes:
            c_class = lp_f_class(coff, doff, qref, propulsion_type)

            build_lp_former_classes.append(c_class)

        # Build lp latter class

        for lp_l_class in lp_latter_classes:
            c_class = lp_l_class(coff, doff, qref, propulsion_type)

            build_lp_latter_classes.append(c_class)

    if elec_shaft_classes != None:

        # Consider fix
        # Build electric class

        for e_class in elec_shaft_classes:
            c_class = e_class(coff, doff, coff_e, qref, qref_e, propulsion_type)

            build_electric_classes.append(c_class)


    return build_hp_classes, build_lp_former_classes, build_lp_latter_classes, build_electric_classes

cdef objective_func(list args):
    cdef:
        np.ndarray[DOUBLE_t, ndim=2] qoff
        np.ndarray[DOUBLE_t, ndim=1] doff
        np.ndarray[DOUBLE_t, ndim=2] qref
        np.ndarray[DOUBLE_t, ndim=2] qoff_e
        np.ndarray[DOUBLE_t, ndim=2] qref_e
        double fscl
        double g
        str propulsion_type
        double W00_core
        double W00
        double W00_fan
        double GenePower
        double ytaele
        double WF30
        double WF70
        double F00_core
        double F19_core
        double F90
        double F00_fan
        double F19_fan
        double COT
        double TIT
        double sfc_off
        double isp_off
        double BPRe
        double FPRe
        double BPR
        double FPR
        double fpr_off
        double OPR
        double thrust_off
        double airflow_off
        list results


    qoff, qoff_e, qref, qref_e, doff, fscl, g, propulsion_type = args

    # The core side of airflow rate
    W00_core = qoff[1, 0]
    # The core side of fan part
    W00 = qoff[1, 20]
    # The distirbuted fan side of airflow rate
    W00_fan = qoff_e[1, 0]

    # Generator Heat Power

    GenePower = 0.0

    if propulsion_type in ['TeDP', 'PartialElectric']:
        ytaele = doff[34]  # electric efficiency
        GenePower = max(qoff_e[6, 10], qref_e[6, 10]) * fscl / ytaele / 1.0e+3

    # Fuel Consumption
    # Combustion Chamber
    WF30 = qoff[0, 30]
    # AfterBurner
    WF70 = qoff[0, 70]

    # Thrust
    F00_core = qoff[0, 0]
    F19_core = qoff[0, 19]
    F90 = qoff[0, 90]
    F00_fan = qoff_e[0, 0]
    F19_fan = qoff_e[0, 19]

    ################################################
    # print('F00_core:', F00_core, 'F19_core:', F19_core, 'F90:', F90, 'F00_fan:', F00_fan, 'F19_fan:', F19_fan)
    ################################################

    # Compressor outlet Temperature
    COT = qoff[4, 30]
    # Turbine Inlet Temperature
    TIT = qoff[4, 40]

    # Specific Fuel Consumption
    sfc_off = WF30 * g * 3600 / (F00_core + F00_fan + F19_core + F19_fan + F90)
    # Specific Thrust
    isp_off = ((F00_core + F00_fan + F19_core + F19_fan + F90) / g) / (W00_core + W00_fan)

    # BPRe, FPRe (Distributed fan)
    BPRe = W00_fan / W00_core
    FPRe = qoff_e[8, 10]

    # BPR, FPR (Core Engine)
    BPR = W00_core / W00
    FPR = doff[32]
    fpr_off = FPR

    # OPR(Overall Pressure Ratio)
    OPR = 0.0
    if propulsion_type in ['turbojet', 'turboshaft', 'TeDP']:
        OPR = qoff[8, 20] * qoff[8, 25]
    elif propulsion_type in ['turbofan', 'PartialElectric']:
        OPR = qoff[8, 10] * qoff[8, 20] * qoff[8, 25]

    # Thrust
    thrust_off = isp_off * (W00_core + W00_fan) * g * fscl


    # Airflow rate
    airflow_off = isp_off * (W00_core + W00_fan)

    results = [GenePower, COT, TIT, sfc_off, isp_off, fpr_off, thrust_off, airflow_off]

    print('=' * 10 + 'Design Off Calculation Results' + '=' * 10)
    print('SFC:', sfc_off, 'TIT_off:', TIT, 'COT_off:', COT)
    print('BPRe_off:', BPRe, 'FPRe:', FPRe)
    print('BPR off:', BPR, 'FPR off:', FPR, 'OPR off:', OPR)
    print('Specific Thrust off:', isp_off)
    print('Thrust off:', thrust_off, 'Airflow rate:', airflow_off)
    print('Generator Heat Power:', GenePower)
    print('=' * 50)
    print('')

    return results