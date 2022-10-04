import numpy as np
import json
from engine_weight import EngineWeight
from thermal_doff import CalcDesignOffPoint
from constraints import EnvConstraint
from design_variable import DesignVariable

# (400, 400, 480, 480) => {"LPC": 0.17327, "HPC": 0.3591, "HPT": 1.43795, "LPT": 1.74665}
# (400, 400, 500, 500) => {"LPC": 0.17327, "HPC": 0.3591, "HPT": 1.2952, "LPT": 1.5756}
# iwasaki => L/D = 16, engine scale = 1.14023340657334  {"LPC": 0.1879156, "HPC": 0.3724552307, "HPT": 1.6546295504, "LPT": 1.4654360576}

class EngineTuning(object):

    def __init__(self, aircraft_name, engine_name, aircraft_type, propulsion_type, off_params, data_base_args):
        self.determined_rev_args = None
        self.aircraft_name = aircraft_name
        self.engine_name = engine_name
        self.aircraft_type = aircraft_type
        self.propulsion_type = propulsion_type
        self.off_params = off_params
        self.data_base_args = data_base_args
        self.aircraft_data_path, self.engine_data_path, self.mission_data_path = data_base_args

        # constraint class
        self.env_const = EnvConstraint()

        # design variable class
        self.design_variable = DesignVariable(self.propulsion_type, self.aircraft_type)

        # components of engine
        self.engine_components_dict = {'LPC': 0, 'HPC': 1, 'HPT': 2, 'LPT': 3}

        # load aircraft and engine data
        f = open(self.aircraft_data_path, 'r')
        self.aircraft_data_file = json.load(f)[self.aircraft_name]
        f.close()

        f = open(self.engine_data_path, 'r')
        self.engine_data_file = json.load(f)[self.engine_name]
        f.close()

    def run_stage_tuning(self):

        target_thermal_design_variables = self.engine_data_file['design_variable']
        self.design_variable.si_design_variable_collect = [target_thermal_design_variables]
        self.design_variable.generate_therm_design_variable_collect()
        target_thermal_design_variables = self.design_variable.therm_design_variable_collect[0]

        self.determine_revolving(target_thermal_design_variables)

        for stage_name in self.engine_components_dict.keys():
            self.define_each_stage_coef(stage_name)

    def determine_revolving(self, thermal_design_variables):

        rev_lp = 1.0
        rev_lp_step = 0.01
        rev_fan = 1.0

        # residual value
        resconv = 0.0
        resconvold = 0.0

        # build calc off design point class
        self.calc_off_design_point_class = CalcDesignOffPoint(self.aircraft_name, self.engine_name, self.aircraft_type, self.propulsion_type, thermal_design_variables, self.off_params, self.data_base_args)

        while True:
            rev_args = [rev_lp, rev_fan]

            self.calc_off_design_point_class.run_off(rev_args)

            self.calc_off_design_point_class.objective_func_doff()

            resconv = 1.0 - self.calc_off_design_point_class.TIT / self.env_const.tit

            if abs(resconv) < 1.0e-5:
                print('Determined Revolving Rate:', rev_lp)
                self.determined_rev_args = [rev_lp, rev_fan]
                break

            if resconv * resconvold < 0.0:
                rev_lp_step *= 0.5

            resconvold = resconv

            rev_lp += np.sign(resconv) * rev_lp_step

    def calc_engine_weight(self):
        self.calc_off_design_point_class.run_off(self.determined_rev_args)
        self.calc_off_design_point_class.objective_func_doff()

        # build engine weight class
        self.engine_weight_class = EngineWeight(self.aircraft_name, self.engine_name, self.aircraft_type, self.propulsion_type, self.calc_off_design_point_class, self.engine_data_path)

        self.engine_weight_class.run_engine()

    def define_each_stage_coef(self, stage_name):
        stage_numbers_dict = self.engine_data_file['stage_number']
        target_stage_number = stage_numbers_dict[stage_name]

        # residual value
        resnum = 0.0
        resnumold = 0.0

        # Initialize stage coefficient
        if stage_name in ['LPC', 'HPC']:
            stage_coef = 0.5
        else:
            stage_coef = 2.0

        stage_coef_step = 0.1

        count = 0


        while True:
            # load engine file
            f = open(self.engine_data_path, 'r')
            self.engine_data_file = json.load(f)[self.engine_name]

            self.calc_engine_weight()

            current_stage_number = self.engine_weight_class.stage_numbers[self.engine_components_dict[stage_name]]

            resnum = 1.0 - current_stage_number / target_stage_number

            if abs(resnum) <= 1.0e-6:
                print('{} stage number: {}'.format(stage_name, current_stage_number))
                break

            if count == 300:
                exit()
            if resnum * resnumold < 0.0:
                stage_coef_step *= 0.5

            resnumold = resnum

            stage_coef += -np.sign(resnum) * stage_coef_step

            # change value of dict
            target_stage_coef_dict = self.engine_data_file['stage_coef']
            target_stage_coef_dict[stage_name] = stage_coef
            self.engine_data_file['stage_coef'] = target_stage_coef_dict

            save_dict = {}
            save_dict[self.engine_name] = self.engine_data_file

            # save file
            f = open(self.engine_data_path, 'w')
            json.dump(save_dict, f)
            f.close()

            count += 1


def test():

    # aircraft and engine parameters
    aircraft_name = 'A320'
    engine_name = 'V2500'
    aircraft_type = 'normal'
    propulsion_type = 'turbofan'

    # database parameters
    aircraft_data_path = './DataBase/aircraft_test.json'
    engine_data_path = './DataBase/engine_test.json'
    mission_data_path = './Missions/maxpayload_base.json'

    data_base_args = [aircraft_data_path, engine_data_path, mission_data_path]

    # off design parameters
    off_altitude = 0.0
    off_mach = 0.0
    off_required_thrust = 133000  # [N]

    off_params = [off_altitude, off_mach, off_required_thrust]

    # build engine tuning class
    et = EngineTuning(aircraft_name, engine_name, aircraft_type, propulsion_type, off_params, data_base_args)

    et.run_stage_tuning()


if __name__ == '__main__':
    test()







