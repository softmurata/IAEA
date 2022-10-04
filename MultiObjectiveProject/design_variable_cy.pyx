import numpy as np
cimport numpy as np
cimport cython


class DesignSpace(object):

    def __init__(self):
        # [min_value, max_value]

        # BPR
        self.BPR = [2.0, 12.0]
        # OPR
        self.OPR = [15.0, 60.0]
        # FPR
        self.FPR = [1.4, 1.8]
        # TIT
        self.TIT = [1300, 1500]
        # BPRe
        self.BPRe = [2.0, 15.0]
        # FPRe
        self.FPRe = [1.1, 1.5]
        # dib_alpha
        self.div_alpha = [0.75, 0.8]
        # electric efficiency
        self.nele = [0.9, 0.99]
        # number of distributed fan
        self.Nfan = [7, 11]
        # electric ratio
        self.electric_ratio = [1.0, 5.0]
        # ToDo after implementing Blended Wing Body class, arrange parameter's range
        # aircraft shape (cockpit)
        self.xw = [0.5, 1.5]
        # aircraft shape (floor x)
        self.u1 = [0.3, 0.5]
        # aircraft shape (floor y)
        self.v1 = [0.3, 0.5]
        # aircraft shape (behind floor x)
        self.u2 = [0.3, 0.5]
        # aircraft shape (behind floor y)
        self.v2 = [0.3, 0.5]
        # range of revolve rate
        self.lp_range = [0.9, 1.1]

        self.design_space_list = [self.BPR, self.OPR, self.FPR, self.TIT, self.BPRe, self.FPRe, self.div_alpha,
                                  self.nele, self.Nfan, self.electric_ratio, self.xw, self.u1, self.v1, self.u2, self.v2]


class DesignVariable(DesignSpace):

    def __init__(self, propulsion_type, aircraft_type):

        #######################Overall Dictionaries#######################################
        self.design_variables_dict = {
            "normal": {"turbojet": ['OPR', 'TIT'],
                       "turboshaft": ['OPR', 'TIT'],
                       "turbofan": ['BPR', 'OPR', 'FPR', 'TIT'],
                       "TeDP": ['OPR', 'TIT', 'BPRe', 'FPRe', 'div_alpha', 'nele', 'Nfan'],
                       "PartialElectric": ['BPR', 'OPR', 'FPR', 'TIT', 'BPRe', 'FPRe', 'div_alpha', 'nele', 'Nfan'],
                       # To Do: Not implemented
                       'battery': ['FPRe', 'Nfan'],
                       'hybridturbojet': ['OPR', 'TIT', 'BPRe', 'FPRe', 'nele', 'Nfan', 'electric_ratio'],
                       'hybridturbofan': ['BPR', 'OPR', 'FPR', 'TIT', 'BPRe', 'FPRe', 'nele', 'Nfan',
                                          'electric_ratio']},

            "BWB": {"turbojet": ['OPR', 'TIT', 'xw', 'u1', 'v1', 'u2', 'v2'],
                    "turboshaft": ['OPR', 'TIT', 'xw', 'u1', 'v1', 'u2', 'v2'],
                    "turbofan": ['BPR', 'OPR', 'FPR', 'TIT', 'xw', 'u1', 'v1', 'u2', 'v2'],
                    "TeDP": ['OPR', 'TIT', 'BPRe', 'FPRe', 'div_alpha', 'nele', 'Nfan', 'xw', 'u1', 'v1', 'u2', 'v2'],
                    "PartialElectric": ['BPR', 'OPR', 'FPR', 'TIT', 'BPRe', 'FPRe', 'div_alpha', 'nele', 'Nfan', 'xw',
                                        'u1', 'v1', 'u2', 'v2'],
                    # To Do: Not implemented
                    'battery': ['FPRe', 'Nfan', 'xw', 'u1', 'v1', 'u2', 'v2'],
                    'hybridturbojet': ['OPR', 'TIT', 'BPRe', 'FPRe', 'nele', 'Nfan', 'electric_ratio', 'xw', 'u1', 'v1',
                                       'u2', 'v2'],
                    'hybridturbofan': ['BPR', 'OPR', 'FPR', 'TIT', 'BPRe', 'FPRe', 'nele', 'Nfan', 'electric_ratio',
                                       'xw', 'u1', 'v1', 'u2', 'v2']}
        }

        self.design_variables_names_list = ['BPR', 'OPR', 'FPR', 'TIT', 'BPRe', 'FPRe', 'div_alpha', 'nele', 'Nfan',
                                            'electric_ratio', 'xw', 'u1', 'v1', 'u2', 'v2']

        # Succeed design space class
        super().__init__()

        ######################################################################################

        # propulsion type
        self.propulsion_type = propulsion_type
        # aircraft type
        self.aircraft_type = aircraft_type

        # design_variables_list
        self.design_variable_name = self.design_variables_dict[self.aircraft_type][self.propulsion_type]

        # Design space dictionary
        self.design_space_dict = {}

        for name in self.design_variable_name:
            idx = self.design_variables_names_list.index(name)

            self.design_space_dict[name] = self.design_space_list[idx]

        # Fix design variable
        self.fixied_dict = None

        # Swarm intelligence module design variable
        self.si_design_variable_collect = []

        # Thermal analysis module design variable
        self.therm_design_variable_collect = []

        # Overall exploration module design variable
        self.oe_design_variable_collect = []

    # set design space function
    def set_design_space(self, range_dict):

        for name, range_list in range_dict.items():
            self.design_space_dict[name] = range_list

    # helper function of set_design_variable
    def make_dv_dict(self, dv_list):

        dv_dict = {}

        for name, dv_val in zip(self.design_variables_dict[self.aircraft_type][self.propulsion_type], dv_list):
            dv_dict[name] = dv_val

        return dv_dict

    # for mono normal calculation
    def set_design_variable(self, dv_list):
        """
        :param :dv_list {'BPR':5.8,'OPR':35.0,'FPR':1.5,'TIT':1470}
        """
        dv_dict = self.make_dv_dict(dv_list)

        therm_design_variable = [0 for _ in range(len(self.design_variables_names_list))]

        for dv_name, dv_val in dv_dict.items():
            dv_idx = self.design_variables_names_list.index(dv_name)

            therm_design_variable[dv_idx] = dv_val

        return therm_design_variable

    # for optimization
    # for example, genetic algorithm or PSO
    def generate_si_design_variable_collect(self, individual_num, fixed_dict):

        if fixed_dict:

            for name, val in fixed_dict.items():
                self.design_space_dict[name] = val

        self.draw_design_space()

        self.si_design_variable_collect = []

        for _ in range(individual_num):
            target_dv = [np.random.rand() for _ in range(len(self.design_space_dict))]

            target_dv = self.reverse_norm(target_dv)

            # Fan number is int
            if 'Nfan' in self.design_variable_name:
                nfan_idx = self.design_variable_name.index('Nfan')
                target_dv[nfan_idx] = int(target_dv[nfan_idx])

            self.si_design_variable_collect.append(target_dv)

        # print(self.si_design_variable_collect)

    # set the swarm intelligence design variables
    def set_si_design_variables_collect(self, design_variables_collect):

        self.si_design_variable_collect = design_variables_collect

    # reverse thermal design variables collection from swarm intelligence collection
    def reverse_si_design_variables_collect(self, thermal_design_variables_collect):

        next_si_design_variables_collect = []

        for thermal_design_variables in thermal_design_variables_collect:
            target_si_design_variables = []

            for tdv in thermal_design_variables:
                if tdv != 0:
                    target_si_design_variables.append(tdv)

            next_si_design_variables_collect.append(target_si_design_variables)

        return next_si_design_variables_collect

    # for thermal cycle calculation
    def generate_therm_design_variable_collect(self):

        if len(self.si_design_variable_collect) > 0:

            self.therm_design_variable_collect = []

            for si_dv in self.si_design_variable_collect:

                target_dv = [0 for _ in range(len(self.design_variables_names_list))]

                for dv_name, si_dv_val in zip(self.design_variable_name, si_dv):
                    idx = self.design_variables_names_list.index(dv_name)

                    target_dv[idx] = si_dv_val

                self.therm_design_variable_collect.append(target_dv)

            # print(self.therm_design_variable_collect)

    def make_design_space_for_norm(self):

        # create design space list for normalization
        design_space = []

        for name in self.design_space_dict.keys():
            design_space.append(self.design_space_dict[name])

        return design_space

    # restorization of design variable sets for thermal analysis simulation
    def reverse_norm(self, norm_dv):

        design_space = self.make_design_space_for_norm()

        new_dv = []

        for idx, val in enumerate(norm_dv):

            if type(design_space[idx]) == list:

                new_dv.append(val * (design_space[idx][1] - design_space[idx][0]) + design_space[idx][0])

            else:

                new_dv.append(design_space[idx])

        return new_dv

    # normalization of design variable sets for Swarm intelligence simulation
    def norm(self, dv):

        design_space = self.make_design_space_for_norm()

        # print('design space:', design_space)

        new_dv = []

        for idx, val in enumerate(dv):

            if type(design_space[idx]) == list:

                new_dv.append((val - design_space[idx][0]) / (design_space[idx][1] - design_space[idx][0]))

            else:

                new_dv.append(1)

        return new_dv

    # describe content of design space
    def draw_design_space(self):

        print('')
        print('=' * 5 + 'design_space' + '=' * 5)

        for name, range_list in self.design_space_dict.items():
            print(str(name) + ':', range_list)

        print('=' * 20)
        print('')

    # describe content of design variable
    def draw_design_variable(self, design_variable):

        print('')
        print('=' * 5 + 'design_variable' + '=' * 5)

        print(design_variable)

        print('=' * 20)
        print('')


# test code for class Design Variable
def test():
    propulsion_type = 'TeDP'
    aircraft_type = 'BWB'

    # Define Design Variable class
    dv = DesignVariable(propulsion_type, aircraft_type)

    # make one design variable
    dv_list = [30.0, 1380, 5.0, 1.4, 0.4, 0.99, 5]

    therm_design_variable = dv.set_design_variable(dv_list)

    dv.draw_design_variable(therm_design_variable)

    # generate design variable collection
    # fixied dict
    fixied_dict = {'OPR': 35, 'BPRe': 8.5}

    # individual number
    individual_num = 10

    # generate design variable sets for swarm inteligence optimization
    dv.generate_si_design_variable_collect(individual_num, fixied_dict)

    # generate desugn variable sets for thermal analysis cycle
    dv.generate_therm_design_variable_collect()

    """
    propulsion_type='turbofan'
    aircraft_type='normal'

    #Define Design Variable class
    dv=DesignVariable(propulsion_type, aircraft_type)

    #turbofan
    dv_list=[4.7,30.0,1.66,1380]

    #fixied dict
    fixied_dict={'OPR':35}

    #individual number
    individual_num=10

    dv_dict=dv.make_dv_dict(dv_list)

    therm_design_variable=dv.set_design_variable(dv_dict)

    dv.draw_design_variable(therm_design_variable)

    dv.generate_design_variable_collect(individual_num, fixied_dict)
    """

    """
    #set design_space test
    range_dict={'BPR':[5.0,10.0]}
    dv.draw_design_space()
    dv.set_design_space(range_dict)
    dv.draw_design_space()
    """


if __name__ == '__main__':
    test()
