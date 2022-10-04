import numpy as np


class InitAirParams(object):

    def __init__(self, aircraft_name, init_mission_class, engine_amplitude):
        self.aircraft_name = aircraft_name
        self.init_mission_class = init_mission_class
        self.engine_amplitude = engine_amplitude

        self.module_dict = {1: 'Wing', 2: 'Horizontal tail', 3: 'Vertical tail', 4: 'Fuselage',
                            5: 'Main Landing Gear', 6: 'Nose landing Gear', 7: 'Nacelle', 8: 'Engine Control',
                            9: 'Starter', 10: 'Fuel System', 11: 'Flight Control', 12: 'APU', 13: 'Instrument',
                            14: 'Hydraulics', 15: 'Electric', 16: 'Avionic', 17: 'furnishing', 18: 'Airconditioner',
                            19: 'Anti ice', 20: 'Handling Gear', 21: 'Engine', 22: 'Passenger Equip'}

        # unit change
        self.m_to_ft = 3.28084
        self.kg_to_lb = 2.204621

        # coefficient of volume effect
        self.crootref = None
        self.BWref = None
        self.base_volume_eff = None

        # (1) Wing params
        self.mw = None  # width of main wing
        self.Swref = None  # Main wing area
        self.Scswref = None  # control wing area
        self.Sw = None
        self.Scsw = None
        self.Nz = None  # 終極荷重係数
        self.AR = None  # Aspect ratio
        self.tc = None  # midspan width of main wing thickness
        self.Lambda = None  # tepar ratio
        self.theta = None  # retreating angle of main wing
        self.BW = None  # Wing Span

        # (2) horizontal tail
        self.Khut = None
        self.Fw = None  # Fuselage width horizontal tail intersection
        self.Bh = None  # horizontal tail span [ft]
        self.Sht = None  # horizontal tail area [ft**2]
        self.Ltail = None  # horizontal tail length [ft]
        self.Ky = None
        self.theta_h = None  # retreating angle of horizontal tail
        self.ARh = None  # aspect ratio of horizontal tail
        self.Se = None

        # (3) Vertical tail
        self.Hthv = None
        self.Svt = None  # Vertical tail area [ft**2]
        self.Kz = None
        self.theta_v = None  # retreating angle of vertical tail
        self.ARv = None  # Aspect ratio of vertical tail
        self.tcroot = None  # root of wing thickness

        # (4) Fuselage
        self.Kdoor = None
        self.Klg = None  # coefficient of mounted main gear
        self.lf = None  # length of fuselage [ft]
        self.Sf = None  # wet area of fuselage  [ft**2]
        self.Kws = None
        self.df = None  # width of fuselage [ft]

        # (5) Main Landing Gear
        self.Kmp = None
        operating_weight_empty = self.init_mission_class.empty_weight * self.kg_to_lb
        max_payload_weight = self.init_mission_class.max_payload_weight * self.kg_to_lb
        self.Wl = operating_weight_empty + max_payload_weight  # zerofuel weight
        self.Nl = None  # 終極荷重係数
        self.Lm = None  # Length of main landing gear [ft]
        self.Nmw = None  # number of main wheel
        self.Nmss = None  # number of main gear shock struts
        self.Vstall = 130.0

        # (6) Nose Landing Gear
        self.Knp = None
        self.Ln = None  # length of nose landing gear
        self.Nnw = None  # number of nose landing gear

        # (7) Nacelle
        self.Kng = None
        self.Nlt = None  # nacelle length
        self.Nw = None
        self.Nen = None  # number of engine

        # (8) Engine Control
        self.Lec = None  # length from engine to cock pit

        # (9) Starter

        # (10) Fuel System

        # (11) Flight Control
        self.Nf = None  # functions performed by controls (typically 4-7)
        self.Nm = None  # mechanical functions (typically 0-2)

        # (12) APU

        # (13) Instrument
        self.Kr = None
        self.Ktp = None
        self.Nc = None  # number of crew if UAV 0.5

        # (14) Hydraulics

        # (15) Electric

        # (16) Avionics

        # (17) furnishing

        # (18) Airconditioner

        # (19) anti - ice

        # (20) handling gear

        # (21) engine

        # (22) passenger equipment

        # other indexes
        self.ctip = None
        self.croot = None

        self.set_main_config()

    def set_main_config(self):
        if self.aircraft_name[0] == 'A':
            # (1) Wing params
            self.mw = 111.83
            self.Swref = 1317.5
            self.Scswref = 510.97
            self.Sw = self.Swref * self.engine_amplitude
            self.Scsw = self.Scswref * self.Sw / self.Swref
            self.Nz = 5.25
            self.AR = 9.5
            self.tc = 0.11
            self.Lambda = 0.24
            self.theta = 25.0

            # (2) horizontal tail
            self.Khut = 1.0
            self.Fw = 6.5
            self.Bh = 40.83
            self.Sht = 333.7
            self.Ltail = 59.0
            self.Ky = 17.7
            self.theta_h = 31.0
            self.ARh = 2.0
            self.Se = 116.795

            # (3) Vertical tail
            self.Hthv = 0.0
            self.Svt = 231.4
            self.Kz = 59.0
            self.theta_v = 49.0
            self.ARv = 1.2
            self.tcroot = 0.15

            # (4) Fuselage
            self.Kdoor = 1.06
            self.Klg = 1.12
            self.lf = 117.0875
            self.Sf = 3825.549
            self.Kws = 0.398689  # 0.75 * ((1.0 + 2.0 * self.Lambda) / (1.0 + self.Lambda)) * (Bw * np.tan
            # (25 * wingsweep_theta / lf))
            self.df = 13.0

            # (5) Main landing gear
            self.Kmp = 1.0
            self.Nl = 5.25
            self.Lm = 112.94
            self.Nmw = 4.0
            self.Nmss = 2.0

            # (6) Nose Landing Gear
            self.Knp = 1.0
            self.Ln = 77.05
            self.Nnw = 2.0

            # (7) Nacelle
            self.Kng = 1.017
            self.Nlt = 17.25
            self.Nw = 5.8
            self.Nen = self.init_mission_class.engine_num

            # (8) Engine Control
            self.Lec = 38.0

            # (11) Flight Control
            self.Nf = 20.0
            self.Nm = 0.0
            # wing span  mw: Main wing width  df: fuselage width
            self.BW = (self.mw - self.df) * np.sqrt(self.Sw / self.Swref) + self.df
            # (13) Instrument
            self.Kr = 1.0
            self.Ktp = 1.0
            self.Nc = 2.0

            # other indexes
            self.croot = self.Sw / self.BW / (1.0 + self.Lambda)
            self.ctip = self.croot * self.Lambda

            # baseline
            self.BWref = self.mw
            self.crootref = self.Swref / self.BWref / (1.0 + self.Lambda)
            self.base_volume_eff = (self.crootref / self.lf) ** 2

