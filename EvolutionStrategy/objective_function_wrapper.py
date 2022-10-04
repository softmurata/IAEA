import json
import argparse
import importlib
import sys
sys.path.append('Function')


# I would like to change registration format at objective function
class FunctionDataBase(object):

    def __init__(self):
        self.function_database_path = 'function_database.json'
        f = open(self.function_database_path)
        self.json_format = json.load(f)
        # show content
        # self.view()

    def load_module(self, function_name, args):
        function_file_name = self.json_format[function_name]
        module = importlib.import_module(function_file_name)
        # print(module.__name__)
        # create test function class
        objective_func_class = module.MainFunc(args)
        design_variables_class = module.MainFuncDesignVariableController(args)
        return objective_func_class, design_variables_class

    def view(self):
        print('current registration condition')
        for key, value in self.json_format.items():
            print('function name: {}   file name: {}'.format(key, value))

    def register(self, function_name, function_file_name):
        f = open(self.function_database_path, 'r')
        try:
            dic = json.load(f)
            sample = {function_name: function_file_name}
            dic.update(sample)
        except:
            # if loading json file fails, initialize dictionary
            dic = {function_name: function_file_name}
        f.close()

        # update json format database
        self.json_format = dic
        # write
        f = open(self.function_database_path, 'w')
        json.dump(dic, f)


class ObjectiveFunctionWrapper(object):

    def __init__(self):
        # Function DataBase class
        self.function_database_class = FunctionDataBase()
        # set objective function and design variables controller
        self.objective_func_class = None
        self.design_variables_class = None

    def register_function(self, function_name, function_file_name):

        self.function_database_class.register(function_name, function_file_name)

    def view_function_in_db(self):

        self.function_database_class.view()

    def select_function_class(self, args):

        objective_func_class, design_variables_class = self.function_database_class.load_module(args.function_name, args)

        self.objective_func_class = objective_func_class
        self.design_variables_class = design_variables_class

    def fitness(self, design_variables):
        # ToDo: we have to revise afterward
        objective_values = []

        for design_variable in design_variables:
            obj = self.objective_func_class.calculate(design_variable)
            objective_values.append(obj)

        return objective_values
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NSGA2 parameters')
    parser.add_argument('--population_num', type=int, default=1000)
    parser.add_argument('--generation_num', type=int, default=5)
    parser.add_argument('--optimize_direction', type=str, default='down_left',
                        help='down_left or upper_right')
    parser.add_argument('--objective_num', type=int, default=2,
                        help='the number of objective functions [y1, y2, ...yn]')
    parser.add_argument('--design_variables_num', type=int, default=3,
                        help='the number of design variables [x1, x2, ... xm]')
    parser.add_argument('--bounds', type=str, default='0.5-1.5/3.5-4.5/2.5-5.5')
    parser.add_argument('--function_name', type=str, default='test', help='objective function name')
    parser.add_argument('--function_file_name', type=str, default='test_func',
                        help='objective function file path, please create function file under Function directory')
    args = parser.parse_args()

    bounds = [[float(bi) for bi in b.split('-')] for b in args.bounds.split('/')]
    args.bounds = bounds

    ofw = ObjectiveFunctionWrapper()

    ofw.register_function(args.function_name, args.function_file_name)

    ofw.select_function_class(args)

    ofw.view_function_in_db()

    # ofw.fitness(design_variables)



