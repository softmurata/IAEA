"""
Note:
    If you want to use original function,
    you should make function file in Function directory and register function name and function file name

Format:
    class MainFuncDesignVariableController():
         def __init__():

         def convert_function_space_into_optimization_space():
             return design variables for optimization space

         def convert_optimization_space_into_function_space():
             return design variables for function space

    class MainFunc():
         def __init__():


         def calculate(self, x):
             return objectives

"""

# for test, this class is implemented in ga_utils.py
class Variable(object):

    def __init__(self, name, norm_value, min_value, max_value):
        self.name = name
        self.value = None
        self.min = min_value
        self.max = max_value
        self.norm_value = norm_value

    def normalize(self):
        if self.value is not None:
            self.norm_value = (self.value - self.min) / (self.max - self.min)

    def denormalize(self):
        if self.norm_value is not None:
            self.value = self.min + self.norm_value * (self.max - self.min)


class MainFuncDesignVariableController(object):
    """
    NOte:
        :param:
          args:
             design_variables_num: the number of design variables(input parameters)  [x1, x2, ..., xm]
             bounds: boundary condition of design variables [[min_num, max_num] for _ in range(design_variables_num)]
    """

    def __init__(self, args):
        self.design_variables_num = args.design_variables_num
        # bounds = [[min, max], [min, max], ..., ]

    def convert_function_space_into_optimization_space(self, func_design_variables):

        optim_design_variables = []

        # normalize
        for func_dv_unit in func_design_variables:
            for idx, variable_class in enumerate(func_dv_unit):
                variable_class.normalize()
                func_dv_unit[idx] = variable_class

            optim_design_variables.append(func_dv_unit)

        return optim_design_variables


    def convert_optimization_space_into_function_space(self, optim_design_variables):

        func_design_variables = []

        # denormalize
        for optim_dv_unit in optim_design_variables:
            for idx, variable_class in enumerate(optim_dv_unit):
                variable_class.denormalize()
                optim_dv_unit[idx] = variable_class

            func_design_variables.append(optim_dv_unit)

        return func_design_variables


class MainFunc(object):
    """
    Note:
        :param
          args: objective_num
                design_variables_num
                bounds
    """
    def __init__(self, args):
        self.name = 'test'
        self.objective_num = args.objective_num
        self.design_variables_num = args.design_variables_num

    def calculate(self, x):

        return x ** 2




def test():
    import argparse
    import random
    import numpy as np
    # define argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--objective_num', type=int, default=2)
    parser.add_argument('--design_variables_num', type=int, default=4)
    parser.add_argument('--individual_num', type=int, default=10)
    args = parser.parse_args()

    # visualize function
    def visualize(dv_type, design_variables_collection):

        vis = []
        for odvc in design_variables_collection:
            dv = []
            for v in odvc:
                if dv_type == 'optimization':
                    vl = v.norm_value
                else:
                    vl = v.value
                dv.append(vl)
            vis.append(dv)
        print(vis)

    # bounds
    bounds = {'x1': [10, 20], 'x2': [5, 30], 'x3': [4, 15], 'x4': [12, 20]}

    # create initial design variable collection
    optim_design_variables_collection = []
    for i in range(args.individual_num):
        dv_unit = []
        for name, bound in bounds.items():
            min_value, max_value = bound
            norm_value = np.random.rand()
            variable = Variable(name, norm_value, min_value, max_value)
            dv_unit.append(variable)
        optim_design_variables_collection.append(dv_unit)

    dv_type = 'optimization'
    visualize(dv_type, optim_design_variables_collection)

    design_variable_class = MainFuncDesignVariableController(args)

    # convert function space into desgin variables
    func_design_variables_collection = design_variable_class.convert_optimization_space_into_function_space(optim_design_variables_collection)
    print('---result check---')
    dv_type = 'optimization'
    visualize(dv_type, func_design_variables_collection)
    dv_type = 'function'
    visualize(dv_type, func_design_variables_collection)
    # objective_func_class = MainFunc(args)


if __name__ == '__main__':
    test()


