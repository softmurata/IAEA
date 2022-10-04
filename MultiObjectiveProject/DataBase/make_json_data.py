import json

"""
Reference website for following public values:
Airbus
    edoc.pub/download/all-about-airbus-a-320-family-pdf-free

Boeing
    Airport Compatibility - Airplane Characteristics for Airport Plannning
"""
# unit
# 1 [inch] = 0.0254 [m]

# helper function

# load data
def load_required_data(isinair, isinengine):
    # data indexes asking for creating aircraft data
    air_required_indexes = ['max_takeoff_weight', 'zerofuel_weight', 'operating_weight_empty', 'cargo_weight',
                            'passenger_num', 'lift_by_drag', 'range', 'altitude', 'mach', 'overall_length',
                            'wide_length', 'height', 'fuselage_wide_length']
    engine_required_indexes = ['engine_weight', 'generation', 'engine_num', 'stage_number', 'fueltype',
                               'required_thrust_ground', 'engine_diameter', 'engine_length', 'design_variable',
                               'sfc_cruise', 'sfc_ground', 'stage_coef']

    # Describe current input data
    def current_input(index):

        print('please input {}'.format(index))

    # return dictionary
    air_dict_data = {}
    engine_dict_data = {}

    if not isinair:
        # input using by standard input
        for index in air_required_indexes:
            current_input(index)
            if index in ['mach', 'overall_length', 'wide_length', 'height', 'fuselage_length']:
                air_dict_data[index] = float(input())
            else:
                air_dict_data[index] = int(input())

    if not isinengine:

        # input engine data
        engine_stage_name = ['Fan', 'LPC', 'HPC', 'HPT', 'LPT']
        engine_dv_name = ['BPR', 'OPR', 'FPR', 'TIT']
        for index in engine_required_indexes:
            # describe current input index
            current_input(index)

            if index in ['stage_number', 'stage_coef']:
                # each stage number inserts
                target_engine_dict = {}
                for esn in engine_stage_name:
                    current_input(esn)
                    target_engine_dict[esn] = float(input())

                engine_dict_data[index] = target_engine_dict

            elif index in ['design_variable']:

                target_dv_dict = {}

                for dvn in engine_dv_name:
                    current_input(dvn)
                    target_dv_dict[dvn] = float(input())

                engine_dict_data[index] = target_dv_dict

            elif index in ['fueltype']:

                engine_dict_data[index] = str(input())

            elif index in ['engine_diameter', 'engine_length', 'sfc_cruise', 'sfc_ground']:

                engine_dict_data[index] = float(input())

            else:
                engine_dict_data[index] = int(input())

    return air_dict_data, engine_dict_data


# calculate lpshaft pressure by hpshaft pressure
def calc_lp_hp_dist(stage_number_dict):
    # initialize variables
    lp_hp_dist_ratio = 0
    # lpshaft stage number,hpshaft stage number
    stage_lp, stage_hp = 0, 0

    # calculate lpshaft stage number and hpshaft stage number
    for esn, esv in stage_number_dict.items():

        if esn == 'LPC':

            stage_lp += esv

        elif esn == 'HPC':

            stage_hp += esv

    lp_hp_dist_ratio = stage_lp / (stage_lp + stage_hp) / 2

    return lp_hp_dist_ratio


# judge techinical level
def judge_techinical_level(generation):
    # minimum and maximum techinical leve;
    min_tech_lev, max_tech_lev = 1, 5
    # minimum and maximum gene
    min_gene, max_gene = 1940, 2030
    # initialize tech8inical level
    tech_lev = 0

    if generation < min_gene:
        tech_lev = min_tech_lev

    elif generation > max_gene:
        tech_lev = max_tech_lev

    else:

        tech_lev = min_tech_lev + (max_tech_lev - min_tech_lev) / (max_gene - min_gene) * (generation - min_gene)

    return tech_lev


# return lhv according to fueltype
def return_lhv(fueltype):
    lhv = 0

    if fueltype == 'Oil':
        lhv = 43000e+3

    elif fueltype == 'Hydrogen':
        lhv = 100000e+3

    return lhv


# combine two dict
def combine_dict(dicts):
    new_dict = {}

    for dict_ in dicts:

        for key, val in dict_.items():
            new_dict[key] = val

    return new_dict


#####################################################################################################################################


# main
def make_json_file(aircraft_name, engine_name, aircraft_data_path='aircraft_test.json',
                   engine_data_path='engine_test.json'):
    new_aircraft_data = {}
    new_engine_data = {}
    # base json data
    json_file = open(aircraft_data_path, 'r')
    base_air_dict = json.load(json_file)

    json_file = open(engine_data_path, 'r')
    base_engine_dict = json.load(json_file)

    # confirm whether or not name is in
    isinair = False
    isinengine = False

    if aircraft_name in base_air_dict.keys():
        isinair = True

    if engine_name in base_engine_dict.keys():
        isinengine = True

    print('')
    print('Fron now on {0} and {1} data input'.format(aircraft_name, engine_name))
    print('')

    air_dict_data, engine_dict_data = load_required_data(isinair, isinengine)

    # data matching
    if isinair and isinengine:
        exit()

    elif isinair and not isinengine:

        air_dict_data = base_air_dict[aircraft_name]

    elif not isinair and isinengine:

        engine_dict_data = base_engine_dict[engine_name]

    # maxpayload
    maxpayload = air_dict_data['zerofuel_weight'] - air_dict_data['operating_weight_empty']

    # maxFuel
    maxfuel = air_dict_data['max_takeoff_weight'] - air_dict_data['zerofuel_weight']

    # Airframe
    airframe = air_dict_data['operating_weight_empty'] - engine_dict_data['engine_weight']

    # required thrust for cruise
    g = 9.81
    required_thrust = (air_dict_data['max_takeoff_weight'] - maxfuel * 0.59) / (air_dict_data['lift_by_drag']) / \
                      engine_dict_data['engine_num'] * g

    # lp_hp_dist_ratio
    stage_number_dict = engine_dict_data['stage_number']

    lp_hp_dist_ratio = calc_lp_hp_dist(stage_number_dict)

    print('LPHPRATIO:', lp_hp_dist_ratio)

    # techinical level
    generation = engine_dict_data['generation']

    tech_lev = judge_techinical_level(generation)

    # LHV
    fueltype = engine_dict_data['fueltype']

    lhv = return_lhv(fueltype)

    # make new dictionary

    # aircraft
    target_air_dict = {}
    target_air_dict['aircraft_weight'] = airframe
    target_air_dict['zerofuel_weight'] = air_dict_data['zerofuel_weight']
    target_air_dict['fuel_weight'] = maxfuel
    target_air_dict['max_payload'] = maxpayload
    target_air_dict['max_takeoff_weight'] = air_dict_data['max_takeoff_weight']
    target_air_dict['operating_weight_empty'] = air_dict_data['operating_weight_empty']
    target_air_dict['cargo_weight'] = air_dict_data['cargo_weight']
    target_air_dict['passenger_num'] = air_dict_data['passenger_num']
    target_air_dict['lift_by_drag'] = air_dict_data['lift_by_drag']
    target_air_dict['range'] = air_dict_data['range']
    target_air_dict['altitude'] = air_dict_data['altitude']
    target_air_dict['mach'] = air_dict_data['mach']
    target_air_dict['overall_length'] = air_dict_data['overall_length']
    target_air_dict['wide_length'] = air_dict_data['wide_length']
    target_air_dict['height'] = air_dict_data['height']
    target_air_dict['fuselage_wide_length'] = air_dict_data['fuselage_wide_length']

    new_aircraft_data[aircraft_name] = target_air_dict

    # engine
    target_engine_dict = {}
    target_engine_dict['engine_weight'] = engine_dict_data['engine_weight']
    target_engine_dict['engine_num'] = engine_dict_data['engine_num']
    target_engine_dict['required_thrust'] = required_thrust
    target_engine_dict['stage_number'] = engine_dict_data['stage_number']
    target_engine_dict['lp_hp_dist_ratio'] = lp_hp_dist_ratio
    target_engine_dict['generation'] = engine_dict_data['generation']
    target_engine_dict['tech_lev'] = tech_lev
    target_engine_dict['fueltype'] = engine_dict_data['fueltype']
    target_engine_dict['lhv'] = lhv
    target_engine_dict['required_thrust_ground'] = engine_dict_data['required_thrust_ground']
    target_engine_dict['engine_diameter'] = engine_dict_data['engine_diameter']
    target_engine_dict['engine_length'] = engine_dict_data['engine_length']
    target_engine_dict['sfc_cruise'] = engine_dict_data['sfc_cruise']
    target_engine_dict['sfc_ground'] = engine_dict_data['sfc_ground']
    target_engine_dict['stage_coef'] = engine_dict_data['stage_coef']

    new_engine_data[engine_name] = target_engine_dict

    if not isinair:
        # combine two dictionaries
        dicts = [base_air_dict, new_aircraft_data]
        new_dict = combine_dict(dicts)

        # write aircraft data in aircraft json file
        json_file = open(aircraft_data_path, 'w')
        json.dump(new_dict, json_file)

    if not isinengine:
        # combine two dictionaries
        dicts = [base_engine_dict, new_engine_data]
        new_dict = combine_dict(dicts)

        # write engine data in engine json file
        json_file = open(engine_data_path, 'w')
        json.dump(new_dict, json_file)


# add json file to the lacking data
def add_lack_data(aircraft_name, engine_name, lack_index_name, lack_index_val, aircraft_data_path, engine_data_path):
    aircraft_f = open(aircraft_data_path, 'r')
    engine_f = open(engine_data_path, 'r')

    aircraft_json_file = json.load(aircraft_f)
    engine_json_file = json.load(engine_f)

    # set the target dict
    target_aircraft_dict = aircraft_json_file[aircraft_name]
    target_engine_dict = engine_json_file[engine_name]

    # Initialize new dict
    new_aircraft_dict = {}
    new_engine_dict = {}

    insert_type, lack_name = lack_index_name.split('-')

    if insert_type == 'aircraft':
        # add new key and value
        target_aircraft_dict[lack_name] = lack_index_val

        for key, val in aircraft_json_file.items():
            if key == aircraft_name:
                new_aircraft_dict[key] = target_aircraft_dict
            else:
                new_aircraft_dict[key] = val

        json_file = open(aircraft_data_path, 'w')
        json.dump(new_aircraft_dict, json_file)

    elif insert_type == 'engine':
        # add new key and value
        target_engine_dict[lack_name] = lack_index_val

        for key, val in engine_json_file.items():
            if key == engine_name:
                new_engine_dict[key] = target_engine_dict
            else:
                new_engine_dict[key] = val

        json_file = open(engine_data_path, 'w')
        json.dump(new_engine_dict, json_file)


if __name__ == '__main__':
    print('Please input arrange type Overall or Add')
    arrange_type = str(input())
    print('Now aircraft name:')
    aircraft_name = str(input())
    print('Now engine name')
    engine_name = str(input())
    # DataBase path
    aircraft_data_path = 'aircraft_test.json'
    engine_data_path = 'engine_test.json'

    if arrange_type == 'Overall':
        make_json_file(aircraft_name, engine_name, aircraft_data_path, engine_data_path)

    elif arrange_type == 'Add':
        # Initialize additional indexes
        lack_names = []

        while True:
            print(
                "Please input index name you want to insert like 'engine-engine_diameter' or 'aircraft-yaw_moment' and so on")
            print('Please input "finish" in order to quit insert option ')
            target_name = str(input())

            if target_name == 'finish':
                break
            lack_names.append(target_name)

        for lack_index_name in lack_names:
            target_name = lack_index_name.split('-')[1]
            print('Please input {} val'.format(target_name))

            lack_index_val = 0

            if target_name == 'design_variable':
                dv_list = []
                dv_name = ['BPR', 'OPR', 'FPR', 'TIT']

                for dvn in dv_name:
                    print('now {} value'.format(dvn))
                    dvv = float(input())
                    dv_list.append(dvv)
                lack_index_val = dv_list
            else:
                lack_index_val = float(input())

            add_lack_data(aircraft_name, engine_name, lack_index_name, lack_index_val, aircraft_data_path,
                          engine_data_path)
