import numpy as np
import json


class InitMission(object):

    def __init__(self, aircraft_name='A320', engine_name='V2500', aircraft_data_path='./DataBase/aircraft.json',
                 engine_data_path='./DataBase/engine.json'):
        json_file = open(aircraft_data_path, 'r')
        # aircraft data json file
        self.aircraft_data = json.load(json_file)[aircraft_name]

        json_file = open(engine_data_path, 'r')
        # aircraft data json file
        self.engine_data = json.load(json_file)[engine_name]

        # passenger weight including baggage
        self.passenger_unit_weight = 90  # [kg]

        # kaili=>km
        self.unit_change = 1.83

        # impact factor coefficient (variables which can change)
        self.cargo_coef = 1
        self.passenger_coef = 1

        # default

        # aircraft
        # weight[kg]
        self.aircraft_weight = None
        self.max_payload_weight = None
        self.payload_weight = None
        self.fuel_weight = None
        self.empty_weight = None
        self.max_takeoff_weight = None
        self.Lift_by_Drag = None
        self.altitude = None  # [m]
        self.mach = None
        self.range = None  # [km]
        self.cargo = None  # [kg]
        self.passenger_num = None
        self.passenger_weight = None

        # engine
        self.engine_weight = None
        self.engine_num = None
        self.required_thrust = None  # [N]
        self.required_thrust_ground = None  # [N]
        self.tech_lev = None
        self.lp_hp_dist_ratio = None
        self.lhv = None
        self.sfc_cruise = None
        self.sfc_ground = None

        self.electric_component_density = (2.5 + 8.3) / (2.5 * 8.3)  # Electric energy density

        self.fuelburn_coef = None

        self.mass_ratio = [0.995, 0.99, 0.98379, 0.0, 0.990, 0.995]

        # baseline mission data path
        self.baseline_mission_data_path = './Missions/maxpayload_base.json'

        # load baseline data from mission file
        self.load_mission_config(self.baseline_mission_data_path)

        # target value of switching point for max takeoff weight
        self.maxpayload_fuelburn = self.fuel_weight

    def set_maxpayload_mission(self, design_point):
        mission_data_path = './Missions/maxpayload_base.json'
        range = self.aircraft_data['range']
        max_takeoff_weight = self.aircraft_data['max_takeoff_weight']
        fuel_weight = self.aircraft_data['fuel_weight']
        passenger_num = self.aircraft_data['passenger_num']
        payload_weight = self.aircraft_data['max_payload']
        cargo_weight = self.aircraft_data['cargo_weight']
        aircraft_weight = self.aircraft_data['aircraft_weight']
        empty_weight = self.aircraft_data['operating_weight_empty']
        lift_by_drag = self.aircraft_data['lift_by_drag']

        if design_point == 'ground':
            altitude = 0
            mach = 0
        else:
            altitude = self.aircraft_data['altitude']
            mach = self.aircraft_data['mach']

        engine_weight = self.engine_data['engine_weight']
        engine_num = self.engine_data['engine_num']

        if design_point == 'ground':
            required_thrust = self.engine_data['required_thrust_ground']
            required_thrust_ground = self.engine_data['required_thrust']
        else:
            required_thrust = self.engine_data['required_thrust']
            required_thrust_ground = self.engine_data['required_thrust_ground']

        tech_lev = self.engine_data['tech_lev']
        lp_hp_dist_ratio = self.engine_data['lp_hp_dist_ratio']
        lhv = self.engine_data['lhv']
        sfc_cruise = self.engine_data['sfc_cruise']
        sfc_ground = self.engine_data['sfc_ground']
        electric_component_density = self.electric_component_density
        fuelburn_coef = self.fuelburn_coef

        # names and values
        file_index_names = ["range", "max_takeoff_weight", "fuel_weight", "passenger_num", "payload_weight",
                            "cargo_weight", "aircraft_weight", "empty_weight", "lift_by_drag", "altitude", "mach",
                            "engine_weight", "engine_num", "required_thrust", "required_thrust_ground", "tech_lev",
                            "lp_hp_dist_ratio", "lhv", "sfc_cruise", "sfc_ground", "electric_component_density",
                            "fuelburn_coef"]

        file_index_values = [range, max_takeoff_weight, fuel_weight, passenger_num, payload_weight,
                             cargo_weight, aircraft_weight, empty_weight, lift_by_drag, altitude, mach,
                             engine_weight, engine_num, required_thrust, required_thrust_ground, tech_lev,
                             lp_hp_dist_ratio, lhv, sfc_cruise, sfc_ground, electric_component_density, fuelburn_coef]

        f = open(mission_data_path, 'w')
        input_file = {}
        for name, value in zip(file_index_names, file_index_values):
            input_file[name] = value

        json.dump(input_file, f)
        f.close()



    def cargo_delta_r(self, max_takeoff_weight):
        # 2d fitting
        coef = [-7.56902238e-06, 1.22411333e+00, -4.19277504e+04]

        num = [max_takeoff_weight ** idx * c for idx, c in enumerate(coef[::-1])]

        return abs(sum(num) * 1000)

    def passenger_delta_r(self, passenger_weight):
        # 1d fitting
        coef = [0.05890909, 45.96363636]

        num = [passenger_weight ** idx * c for idx, c in enumerate(coef[::-1])]

        return abs(sum(num))

    def set_mission(self, mission_coef_args):
        """

        :param mission_coef_args: [cargo_coef (0 ~ 1) ,passenger_coef (0 ~ 1)]
        :param mission_data_path: str
        :return:
        """
        # set variables (coefficients of cargo or passenger)
        cargo_coef, passenger_coef = mission_coef_args

        self.cargo_coef, self.passenger_coef = cargo_coef, passenger_coef
        # diminish range

        # factor cargo
        self.max_takeoff_weight = self.aircraft_data['max_takeoff_weight']
        self.max_cargo = self.aircraft_data['cargo_weight']
        self.cargo = self.max_cargo * (self.cargo_coef)
        product_cargo_delta_r = self.cargo_delta_r(self.max_takeoff_weight)

        max_delta_range_for_cargo = product_cargo_delta_r / self.max_cargo

        delta_range_for_cargo = (max_delta_range_for_cargo * (1.0 - self.cargo_coef))

        # factor passenger
        self.passenger_num = self.aircraft_data['passenger_num']
        self.passenger_num = int(self.passenger_num * self.passenger_coef)
        self.passenger_weight = self.passenger_unit_weight * self.passenger_num
        max_delta_range_for_pas = self.passenger_delta_r(
            self.passenger_unit_weight * self.aircraft_data['passenger_num'])

        delta_range_for_pas = max_delta_range_for_pas * (1.0 - self.passenger_coef)

        # total diminishment of range
        delta_range = (delta_range_for_cargo + delta_range_for_pas) * self.unit_change

        print('delta range cargo:', delta_range_for_cargo)
        print('delta range for passenger:', delta_range_for_pas)

        # total payload weight
        self.payload_weight = self.cargo + self.passenger_weight
        self.max_payload_weight = self.aircraft_data['max_payload']

        # mission range
        self.range = self.aircraft_data['range'] + delta_range
        print('current mission range[km]:', self.range)
        # self.range = 4530

        # other index
        self.aircraft_weight = self.aircraft_data['aircraft_weight']
        self.engine_weight = self.engine_data['engine_weight']
        self.engine_num = self.engine_data['engine_num']
        self.fuel_weight = self.max_takeoff_weight - (
                self.aircraft_weight + self.engine_weight * self.engine_num + self.payload_weight)

        self.empty_weight = self.aircraft_data['operating_weight_empty']

        print('cargo weight[kg]:', self.cargo)
        print('passenger number:', self.passenger_num)
        print('payload weight[kg]:', self.payload_weight, 'max payload[kg]:', self.aircraft_data['max_payload'])
        print('fuel_weight[kg]:', self.fuel_weight)

        # non calculating mission parameters
        self.Lift_by_Drag = self.aircraft_data['lift_by_drag']
        self.altitude = self.aircraft_data['altitude']
        self.mach = self.aircraft_data['mach']

        # engine part
        self.engine_weight = self.engine_data['engine_weight']
        self.engine_num = self.engine_data['engine_num']
        self.required_thrust = self.engine_data['required_thrust']
        self.required_thrust_ground = self.engine_data['required_thrust_ground']
        self.tech_lev = self.engine_data['tech_lev']
        self.lp_hp_dist_ratio = self.engine_data['lp_hp_dist_ratio']
        self.lhv = self.engine_data['lhv']
        self.sfc_cruise = self.engine_data['sfc_cruise']
        self.sfc_ground = self.engine_data['sfc_ground']


    def target_fuelburn(self, target_fuelburn, mission_data_path):
        mission_coef_args = [1.0, 1.0]

        # In case of loading max passenger and cargo, max takeoff weight has to diminish according to
        # the increment of fuelburn
        if target_fuelburn <= self.maxpayload_fuelburn:
            self.max_takeoff_weight = self.max_takeoff_weight - (self.maxpayload_fuelburn - target_fuelburn)
            self.fuel_weight = target_fuelburn
            self.save_mission_config(mission_data_path)

            return

        fuelburn_diff = 0
        fuelburn_diffold = 0
        coef_step = 0.1
        count = 0

        while True:
            if count == 100:
                break

            self.set_mission(mission_coef_args)
            fuelburn_diff = 1.0 - self.fuel_weight / target_fuelburn

            print('fuelburn diff:', fuelburn_diff)

            if abs(fuelburn_diff) < 1.0e-6:
                self.save_mission_config(mission_data_path)
                break

            # update
            if fuelburn_diff * fuelburn_diffold < 0.0:
                coef_step *= 0.5

            mission_coef_args[0] += -np.sign(fuelburn_diff) * coef_step

            if mission_coef_args[0] <= 0:
                mission_coef_args[0] = 0.0
                mission_coef_args[1] = -np.sign(fuelburn_diff) * coef_step

            fuelburn_diffold = fuelburn_diff

            count += 1

    # load mission_config
    def load_mission_config(self, mission_data_path):
        f = open(mission_data_path, 'r')
        mission_file = json.load(f)
        # aircraft
        # weight[kg]
        self.aircraft_weight = mission_file['aircraft_weight']
        self.max_payload_weight = self.aircraft_data['max_payload']
        self.payload_weight = mission_file['payload_weight']
        self.fuel_weight = mission_file['fuel_weight']
        self.empty_weight = mission_file['empty_weight']
        self.max_takeoff_weight = mission_file['max_takeoff_weight']
        self.Lift_by_Drag = mission_file['lift_by_drag']
        self.altitude = mission_file['altitude']  # [m]
        self.mach = mission_file['mach']
        self.range = mission_file['range']  # [km]  # 4530
        self.cargo = mission_file['cargo_weight']  # [kg]
        self.passenger_num = mission_file['passenger_num']
        self.passenger_weight = self.passenger_unit_weight * self.passenger_num

        # engine
        self.engine_weight = mission_file['engine_weight']
        self.engine_num = mission_file['engine_num']
        self.required_thrust = mission_file['required_thrust']  # [N]
        self.required_thrust_ground = mission_file['required_thrust_ground']  # [N]
        self.tech_lev = mission_file['tech_lev']
        self.lp_hp_dist_ratio = mission_file['lp_hp_dist_ratio']
        self.lhv = mission_file['lhv']
        self.sfc_cruise = mission_file['sfc_cruise']
        self.sfc_ground = mission_file['sfc_ground']

        self.electric_component_density = mission_file['electric_component_density']  # Electric energy density
        self.fuelburn_coef = mission_file['fuelburn_coef']

    # save mission configuration
    def save_mission_config(self, mission_data_path):

        new_mission_dict = {}

        # aircraft part
        new_mission_dict['range'] = self.range
        new_mission_dict['max_takeoff_weight'] = self.max_takeoff_weight
        new_mission_dict['fuel_weight'] = self.fuel_weight
        new_mission_dict['passenger_num'] = self.passenger_num
        new_mission_dict['payload_weight'] = self.payload_weight
        new_mission_dict['cargo_weight'] = self.cargo
        new_mission_dict['aircraft_weight'] = self.aircraft_weight
        new_mission_dict['empty_weight'] = self.empty_weight
        new_mission_dict['lift_by_drag'] = self.Lift_by_Drag
        new_mission_dict['altitude'] = self.altitude
        new_mission_dict['mach'] = self.mach

        # engine part
        new_mission_dict['engine_weight'] = self.engine_weight
        new_mission_dict['engine_num'] = self.engine_num
        new_mission_dict['required_thrust'] = self.required_thrust
        new_mission_dict['required_thrust_ground'] = self.required_thrust_ground
        new_mission_dict['tech_lev'] = self.tech_lev
        new_mission_dict['lp_hp_dist_ratio'] = self.lp_hp_dist_ratio
        new_mission_dict['lhv'] = self.lhv
        new_mission_dict['sfc_cruise'] = self.sfc_cruise
        new_mission_dict['sfc_ground'] = self.sfc_ground
        new_mission_dict['electric_component_density'] = self.electric_component_density
        new_mission_dict['fuelburn_coef'] = self.fuelburn_coef

        f = open(mission_data_path, 'w')

        json.dump(new_mission_dict, f)

        print('=' * 10 + ' Succeed the mission save ' + '=' * 10)


if __name__ == '__main__':
    print('Now aircraft_name')
    aircraft_name = str(input())
    print('Now engine_name')
    engine_name = str(input())
    aircraft_data_path = './DataBase/aircraft_test.json'
    engine_data_path = './DataBase/engine_test.json'

    im = InitMission(aircraft_name, engine_name, aircraft_data_path, engine_data_path)

    im.set_maxpayload_mission(design_point='cruise')

    # mission_coef_args
    mission_coef_args = [1.0, 1.0]
    im.set_mission(mission_coef_args)

    # load mission
    load_mission_data_path = './Missions/cargo{}_passenger{}_.json'.format(mission_coef_args[0], mission_coef_args[1])
    im.load_mission_config(load_mission_data_path)
    print(im.range)
