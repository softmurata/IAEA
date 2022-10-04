class BoundaryCondition(object):
    def __init__(self):
        self.db = {}

    def register(self, name, bound):
        self.db[name] = bound

# aircraft boundary condition class
class AircraftBoundaryCondtion(BoundaryCondition):

    def __init__(self):
        super().__init__()

        # default
        # Main wing
        self.bmw = [40, 80]  # main wing span
        self.cmtip = [3, 10]  # main wing tip chord
        self.thetam = [20, 40]  # retreat angle of main wing
        self.tcmw = [0.07, 0.2]  # the ratio of main wing thickness and main wing chord
        self.armw = [7.0, 10.0]  # aspect ratio of main wing

        # Vertical wing
        self.bvw = [10, 25]  # vertical wing span
        self.cvtip = [2, 6]  # vertical wing tip chord
        self.thetav = [30, 50]  # retreat angle of vertical wing
        self.tcvw = [0.1, 0.25]  # the ratio of vertical wing thickness and vertical wing chord
        self.arvw = [2, 5] # aspect ratio of vertical wing

        # Horizontal wing
        self.bhw = [10, 25]  # horizontal wing span
        self.chtip = [2, 6]  # horizontal wing tip chord
        self.thetah = [20, 40]  # retreat angle of horizontal wing
        self.tchw = [0.1, 0.25]  # the ratio of horizontal wing thickness and horizontal wing chord
        self.arhw = [2, 5]  # aspect ratio of horizontal wing

        # Fuselage
        self.huc = [1.5, 2.5]  # height of upper part of cabin
        self.hlc = [1.5, 2.5]  # height of lower part of cabin
        self.wc = [1.0, 2.0]  # width of cabin
        self.hlf = [1.5, 2.5]  # height of lower part of fuselage
        self.huf = [1.5, 2.5]  # height of upper part of fuselage
        self.wf = [1.5, 2.5]  # width of fuselage
        self.hau = [1.0, 2.0]  # height of upper part of after cabin
        self.wa = [0.1, 0.3]  # width of after cabin
        self.l1 = [5, 15]  # length of cabin
        self.l2 = [20, 30]  # length of fuselage
        self.l3 = [3, 6]  # length of after cabin
        self.uk = [0.3, 0.7]  # control coefficient for bezier curve at cockpit
        # Aircraft configuration parameters
        # main wing
        self.jmx = [0.2, 0.6]  # x coord coefficient of joint point of main wing
        self.jmz = [0, 0]  # z coord coefficient of joint point of main wing
        self.pm = [0.2, 0.6]  # constant for airfoil shape(main wing)

        # horizontal wing
        self.jhx = [0.8, 1.0]  # x coord coefficient of joint point of horizontal wing
        self.jhz = [0, 0]  # z coord coefficient of joint point of horizontal wing
        self.ph = [0.2, 0.6]  # constant for airfoil shape(horizontal wing)

        # vertical wing
        self.jvx = [0.9, 1.0]  # x coord coefficient of joint point of vertical wing
        self.jvz = [0, 0]  # z coord coefficient of joint point of vertical wing
        self.pv = [0.2, 0.6]  # constant for airfoil shape(vertical wing)




# engine boundary condition class
class EngineBoundaryCondition(BoundaryCondition):

    def __init__(self):
        super().__init__()

        # Engine
        # core engine
        # thermodynamic
        self.BPR = [3.0, 20.0]  # bypass ratio
        self.OPR = [10.0, 40.0]  # overall pressure ratio
        self.FPR = [1.2, 2.0]  # fan pressure ratio
        self.TIT = [1200. 1600]  # turbine inlet temperature
        # hydrogen aircraft

        # distributed fan
        self.BPRe = [20.0, 40.0] # bypass ratio of distributed fan
        self.FPRe = [1.2, 1.6]  # fan pressure ratio of distributed fan
        self.alpha = [0.7, 0.9] # distributed ratio of power(core: distributed fan)
        self.ele = [0.9, 0.99]  # electric efficiency
        self.nfan = [3, 20]  # the number of distributed fan

        # shape
        self.fim = [0.4, 0.6]  # fan inlet mach number(0.4 - 0.6)
        self.lcim = [0.4, 0.6]  # lpc(low pressure compressor) inlet mach number(0.4 - 0.6)
        self.hcim = [0.4, 0.6]  # hpc(high pressure compressor) inlet mach number(0.4 - 0.6)
        self.ccim = [0.2, 0.4]  # cc(combustion chamber) inlet mach number(0.2 - 0.4)
        self.htim = [0.05, 0.2]  # hpt(high pressure turbine) inlet mach number(0.05 - 0.2)
        self.htcim = [0.35, 0.55]  # hpt cool inlet mach number(0.35 - 0.55)
        self.ltim = [0.35, 0.55]  # lpt inlet mach number(0.35 - 0.55)
        self.ltcim = [0.35, 0.55]  # lpt cool inlet mach number(0.35 - 0.55)
        self.nim = [0.2, 0.4]  # nozzle inlet mach number(0.2 - 0.4)

        self.lcom = [0.4, 0.6]  # lpc outlet mach number(0.4 - 0.6)
        self.hcom = [0.4, 0.6]  # hpc outlet mach number(0.4 - 0.6)
        self.htom = [0.35, 0.55]  # hpt outlet mach number(0.35 - 0.55)
        self.ltom = [0.35, 0.55]  # lpt outlet mach number(0.35 - 0.55)

        self.fthr = [0.25, 0.45]  # fan tip hub ratio(0.25 - 0.45)
        self.lthr = [0.4, 0.6]  # lpc tip hub ratio(0.4 - 0.6)
        self.hthr = [0.4, 0.6]  # hpc tip hub ratio(0.4 - 0.6)

        # efficiency
        # ToDo: need technical level settings?
        self.epsb = [0.06, 0.06]  # overall pressure loss at burner(0.06)
        self.ytab = [0.96, 0.96]  # efficiency of burner(0.96)
        self.cahd = [0.0, 0.2] # distribution of cool air at high pressure turbine
        self.cald = [0.0, 0.3]  # distribution of cool air at low pressure turbine
        self.mhp = [0.99, 0.99]  # mechanical efficiency of high pressure turbine
        self.mlp = [0.99, 0.99] # mechanical efficiency of low pressure turbine
        self.epsab = [0.0, 0.0]  # overall pressure loss at after burner(0.0)
        self.ytaab = [1.0, 1.0]  # efficiency of after burner(1.0)
        self.paicn = [0.02, 0.02]  # overall pressure loss of core nozzle
        self.paibn = [0.02, 0.02]  # oversll pressure loss of bypass nozzle(fan)

        # performance
        # type: list
        self.stage_number = [[1, 2], [2, 6], [6, 10], [1, 2], [2, 5]]  # stage number of each engine component(fan, lpc, hpc, hpt, lpt)
        self.stage_load_coefficient = [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [2.0, 3.0], [2.0, 3.0]]  # stage load coefficient of each engine component


"""
Note:
    # Mission
    tech_lev
    altitude
    mach number
    passenger num
    mass ratio
        => ex) [0.995, 0.99, 0.98, 0.0, 0.990, 0.995]
    fuel type
    max takeoff weight
        => max takeoff weight = aircraft weight + engine weight * engine num + cargo weight + passenger weight
    required thrust at design point(cruise or takeoff?)
    cruise range(km)
    

    # IAEA design variables configuration
    
    S = (croot + ctip) * b
    AR = b ** 2 / S = b / (croot + ctip) = b / (1 + t) * ctip
    t = b / (AR * ctip) - 1
    thick = ctip * tc
    
    # Main wing
    bmw: main wing span
    cmtip: main wing tip chord
    thetam: retreat angle of main wing
    tcmw: the ratio of main wing thickness and main wing chord
    armw: aspect ratio of main wing
    
    # Vertical wing
    bvw: vertical wing span
    cvtip: vertical wing tip chord
    thetav: retreat angle of vertical wing
    tcvw: the ratio of vertical wing thickness and vertical wing chord
    arvw: aspect ratio of vertical wing
    
    # Horizontal wing
    bhw: horizontal wing span
    chtip: horizontal wing tip chord
    thetah: retreat angle of horizontal wing
    tchw: the ratio of horizontal wing thickness and horizontal wing chord
    arhw: aspect ratio of horizontal wing
    
    # Fuselage
    huc: height of upper part of cabin
    hlc: height of lower part of cabin
    wc: width of cabin
    hlf: height of lower part of fuselage
    huf: height of upper part of fuselage
    wf: width of fuselage
    hau: height of upper part of after cabin
    wa: width of after cabin
    l1: length of cabin
    l2: length of fuselage
    l3: length of after cabin
    uk: control coefficient for bezier curve at cockpit
    
    
    # Aircraft configuration parameters
    # main wing
    jmx: x coord coefficient of joint point of main wing
    jmz: z coord coefficient of joint point of main wing
    pm: constant for airfoil shape(main wing)
    
    # horizontal wing
    jhx: x coord coefficient of joint point of horizontal wing
    jhz: z coord coefficient of joint point of horizontal wing
    ph: constant for airfoil shape(horizontal wing)
    
    # vertical wing
    jvx: x coord coefficient of joint point of vertical wing
    jvz: z coord coefficient of joint point of vertical wing
    pv: constant for airfoil shape(vertical wing)
    
    
    # Engine
    # core engine
    # thermodynamic
    BPR: bypass ratio
    OPR: overall pressure ratio
    FPR: fan pressure ratio
    TIT: turbine inlet temperature
    # hydrogen aircraft
    
    
    # distributed fan
    BPRe: bypass ratio of distributed fan
    FPRe: fan pressure ratio of distributed fan
    alpha: distributed ratio of power(core:distributed fan)
    ele: electric efficiency
    nfan: the number of distributed fan
    
    
    # shape
    fim: fan inlet number(0.4 - 0.6)
    lcim: lpc inlet number(0.4 - 0.6)
    hcim: hpc inlet mach number(0.4 - 0.6)
    ccim: cc inlet mach number(0.2 - 0.4)
    htim: hpt inlet mach number(0.05 - 0.2)
    htcim: hpt cool inlet mach number(0.35 - 0.55)
    ltim: lpt inlet mach number(0.35 - 0.55)
    ltcim: lpt cool inlet mach number(0.35 - 0.55)
    nim: nozzle inlet mach number(0.2 - 0.4)
    
    lcom: lpc outlet mach number(0.4 - 0.6)
    hcom: hpc outlet mach number(0.4 - 0.6)
    htom: hpt outlet mach number(0.35 - 0.55)
    ltom: lpt outlet mach number(0.35 - 0.55)
    
    fthr: fan tip hub ratio(0.25 - 0.45)
    lthr: lpc tip hub ratio(0.4 - 0.6)
    hthr: hpc tip hub ratio(0.4 - 0.6)
    
    # efficiency
    # ToDo: need technical level settings? 
    epsb: overall pressure loss at burner(0.06)
    ytab: efficiency of burner(0.96)
    cahd: distribution of cool air at high pressure turbine
    cald: distribution of cool air at low pressure turbine
    mhp: mechanical efficiency of high pressure turbine
    mlp: mechanical efficiency of low pressure turbine
    epsab: overall pressure loss at after burner(0.0)
    ytaab: efficiency of after burner(1.0)
    paicn: overall pressure loss of core nozzle
    paibn: oversll pressure loss of bypass nozzle(fan)
    
    # performance
    # type: list
    stage number: stage number of each engine component
    stage load coefficient: stage load coefficient of each engine component
    


"""

# create boundary condition's database accroding to aircraft type
def create_iaea_boundary_condition_database():
    """
    :return:
    """

    # aircraft
    aircraft_dv_names = ['bmw',
                         'cmtip',
                         'thetam',
                         '',
                         '',
                         ]

    # engine
    engine_dv_names = []


