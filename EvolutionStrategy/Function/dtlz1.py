# import necessary library
from deap import benchmarks


# Design Variables Controller
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
        self.bounds = args.bounds

    def convert_function_space_into_optimization_space(self, func_design_variables):
        optim_design_variables = []

        # normalize

        return optim_design_variables

    def convert_optimization_space_into_function_space(self, optim_design_variables):
        func_design_variables = []

        # denormalize

        return func_design_variables


# Function class
class MainFunc(object):

    def __init__(self, objective_nums, design_variables_num, bounds):
        self.name = 'test'
        self.objective_nums = objective_nums
        self.design_variables_num = design_variables_num
        self.bounds = bounds


    def calculate(self, x):
        # dtlz1

        return benchmarks.dtlz1(x, self.objective_nums)


