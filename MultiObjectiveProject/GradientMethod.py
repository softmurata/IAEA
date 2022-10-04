import numpy as np
from scipy import optimize
from integration_env_gm import IntegrationEnvGradient

# 数値微分でscipyを使って実際に計算してみる
# => scipyの中でも最適化できるものと出来ないものがある


class GradientTestDesignSpace(object):
    """
    Attributes
    ---------------
    design_variables_num: int
                        the number of design variables
    design_space_list: list
                        the list which contains each boundary condition

    """

    def __init__(self, design_variables_num):
        # the number of design variables
        self.design_variables_num = design_variables_num
        bound = [0, 4]
        self.design_space_list = [bound for _ in range(self.design_variables_num)]

    def set_dv_bound(self, idx, bound):
        """

        :param idx: the index of design variables
        :param bound: the range of boundary condition
        :return: None
        """
        self.design_space_list[idx] = bound


class GradientTestDesignVariable(GradientTestDesignSpace):

    def __init__(self, design_variables_num):
        # succeed the design space class
        super().__init__(design_variables_num)


class GradientTestFunc(object):
    """
    Attributes
    ------------------
    name: str

    number: int
            the number which indicated predetermined function
    design_variables_num: int

    design_variable: class object
            class for setting or modifying something to design variables
    """

    def __init__(self, design_variable_class, design_variables_num):

        self.name = 'gradient test'
        # function index
        self.number = 1
        # design variables num
        self.design_variables_num = design_variables_num
        # define design variable class
        self.design_variable = design_variable_class

        self.set_bounds()

    def set_bounds(self):
        if self.number == 3:
            bounds = [-32, 32]
            for idx in range(self.design_variables_num):
                self.design_variable.set_dv_bound(idx, bounds)
        elif self.number == 4:
            bounds = [-10, 10]
            for idx in range(self.design_variables_num):
                self.design_variable.set_dv_bound(idx, bounds)
        elif self.number == 5:
            bounds = [-5, 5]
            for idx in range(self.design_variables_num):
                self.design_variable.set_dv_bound(idx, bounds)

    def test_function(self, args, number):
        """

        :param args: function variables
        :param number: function number
        :return: function value
        """

        if number == 0:

            return args[0] ** 4 + args[1] ** 3 + args[2] ** 2 + args[3] ** 3 - 2 * args[0] ** 2 * args[1] - 4 * args[0] * args[3]

        elif number == 1:

            return args[0] ** 3 + args[1] ** 3 - 3 * args[0] * args[1]

        elif number == 2:

            return args[0] ** 2 - 2 * (args[0] + args[1] + args[2]) + args[1] ** 2 + args[2] ** 2

        elif number == 3:
            x = np.array(args)
            t1 = 20
            t2 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / len(x)))
            t3 = np.e
            t4 = -np.exp(np.sum(np.cos(2.0 * np.pi * x)) / len(x))

            return t1 + t2 + t3 + t4

        elif number == 4:
            x = np.array(args)

            return np.sum(x ** 2)

        elif number == 5:
            val = 0
            for idx in range(self.design_variables_num - 1):
                t1 = 100 * (args[idx + 1] - args[idx] ** 2) ** 2
                t2 = (args[idx] - 1) ** 2
                val += (t1 + t2)

            return val

    def fitness(self, solution):
        """
        calculate objective value

        :param solution: design variables
        :return: objectives
        """

        objectives = self.test_function(solution, self.number)

        return objectives


# This file includes gradient descent method
# Conjugate gradient method
# format
# __init__, calc_gradient, solve
# ToDo we finished implementing algorithm but against particular functions, this class did not work well.
# ToDO further investigation on various optimal function is necessary

class ConjugateGradientMethod(object):
    """
    Only Single Objective Optimization

    Note:
        Conjugate Gradient Method Algorithm:


    Attributes
    -------------------

    diff_step: float
            the width for numerical differential
    objective_func: class object
            class for outresulting the objective value
    design_variables_num: int
            the number of design variables
    """

    def __init__(self, objective_func, diff_step=0.01):

        # set the value of step for numerical diffusion
        self.diff_step = diff_step
        # class for calculating the objective values
        # the args of function is f(args), so if you use more complicated objective function class,
        # you have to create the wrapper class or function
        self.objective_func = objective_func
        # set the number of design variables
        self.design_variables_num = self.objective_func.design_variables_num

    # calculate gradients
    def calc_gradient(self, pre_solution):

        """
        calculate the derivative of particular function

        :param pre_solution: design variables
        :return: numerical differential
        """
        # set the next solution
        next_solution = []
        for idx in range(self.design_variables_num):
            target_sol = pre_solution[idx] + self.diff_step
            bounds = self.objective_func.design_variable.design_space_list[idx]
            min_val, max_val = bounds
            if target_sol <= min_val:
                target_sol = min_val
            if target_sol >= max_val:
                target_sol = max_val
            next_solution.append(target_sol)

        # no bounds
        # next_solution = [pre_solution[idx] + self.diff_step for idx in range(self.design_variables_num)]

        # partial differential
        arguments_sets = []
        target_idx = 0

        for _ in range(self.design_variables_num):
            # Initialize the new design variables
            target_solution = pre_solution.copy()
            next_target_solution = next_solution.copy()
            # change the value of design variables
            target_solution[target_idx] = next_target_solution[target_idx]
            # add the list called arguments sets
            arguments_sets.append([target_solution, pre_solution])
            # update the target design variables index
            target_idx += 1

        # calculate gradient
        diffs = []

        for idx, arg_set in enumerate(arguments_sets):
            next_args, pre_args = arg_set
            # calculate numerical differential
            diff = (self.objective_func.fitness(next_args) - self.objective_func.fitness(pre_args)) / self.diff_step

            diffs.append(diff)

        return diffs

    # solve the optimization problems
    def solve(self, init_solution):
        """
        conduct optimization of conjugate gradient method

        :param init_solution: initial design variables
        :return: current_solution, current_objective
        """
        # Initialize previous solution
        pre_solution = init_solution

        # Initialize vectors of search direction
        grad_func = self.calc_gradient(pre_solution)
        direction = [-g for g in grad_func]

        # In this method, we have applied to the armijo condition
        # In this materials, another condition which compensates the optimality for this optimization problems.
        # If you want to use other or more strict optimal condition, you have to read its materials and implement the this part

        # armijo coefficients
        armijo_coef = 0.6

        # Initialize calculation count (for error)
        calc_count = 0

        while True:

            if calc_count == 100:
                exit()

            # Line Search
            alpha = 1.0  # line search coefficient
            alpha_step = 0.01
            max_alpha_step = (alpha - 0) / alpha_step

            # line search count
            line_search_count = 0

            # calculate the previous gradient
            pre_grad_func = self.calc_gradient(pre_solution)

            # Line Search Part
            while True:
                derphi0 = np.dot(np.array(direction), np.array(pre_grad_func))
                # for armijo condition
                # set the optimal target value
                armijo_target = self.objective_func.fitness(pre_solution) + armijo_coef * alpha * derphi0
                # line search design variables
                current_solution = [pre_solution[idx] + alpha * direction[idx] for idx in range(self.design_variables_num)]

                # calculate current objectives
                current_objective = self.objective_func.fitness(current_solution)

                # check to meet the optimality
                if current_objective < armijo_target:

                    break

                if line_search_count == 2 * max_alpha_step:

                    break

                # update the line search coefficient
                alpha -= alpha_step
                # if alpha < 0:
                #  alpha = -derphi0 * 1.0 ** 2 / 2.0 / (armijo_target - self.objective_func.fitness(pre_solution) - derphi0 * 1.0)
                line_search_count += 1

            print(pre_solution, current_solution)

            # current gradient function values
            current_grad_func = self.calc_gradient(current_solution)
            # judge convergence
            dir_norm = np.linalg.norm(np.array(direction))  # * alpha
            # residual between current objectives and previous objectives
            res_objective = current_objective - self.objective_func.fitness(pre_solution)
            # confirmation of behaviour of residual
            print('direction norm:', dir_norm)
            print('residual of two objectives:', res_objective)

            if abs(dir_norm) < 1.0e-5:

                return current_solution, current_objective

            # beta
            yk = [current_grad_func[idx] - pre_grad_func[idx] for idx in range(self.design_variables_num)]
            beta = np.dot(np.array(yk), np.array(current_grad_func)) / np.dot(np.array(pre_grad_func), np.array(pre_grad_func))
            print(beta)
            # update direction vector
            direction = [-current_grad_func[idx] + beta * direction[idx] for idx in range(self.design_variables_num)]
            # update count of iteration
            calc_count += 1
            # replace previous solution
            pre_solution = current_solution


# normal gradient method
class GradientDescentMethod(object):
    """
    Only Single Objective Optimization

    Note:
        Gradient Descent Method Algorithm:


    Attributes
    ----------------------

    diff_step: float
            the width for numerical differential
    objective_func: class object
            class for outresulting the objective value
    design_variables_num: int
            the number of design variables

    """

    def __init__(self, objective_func, diff_step=0.01):
        # objective function class
        self.objective_func = objective_func
        # the numerical differential step
        self.diff_step = diff_step
        # the number of design variables
        self.design_variables_num = self.objective_func.design_variables_num

    # numerical differential
    def calc_gradient(self, pre_solution):
        """
        calculate the derivative of particular function

        :param pre_solution: design variables
        :return: numerical differential
        """
        # Including the constraint of design variables, add minute step to the pre solution
        next_solution = []
        for idx in range(self.design_variables_num):
            target_sol = pre_solution[idx] + self.diff_step
            bounds = self.objective_func.design_variable.design_space_list[idx]
            min_val, max_val = bounds
            if target_sol <= min_val:
                target_sol = min_val
            if target_sol >= max_val:
                target_sol = max_val
            next_solution.append(target_sol)

        # no bounds
        # next_solution = [pre_solution[idx] + self.diff_step for idx in range(self.design_variables_num)]

        # partial differential
        arguments_sets = []
        target_idx = 0

        for _ in range(self.design_variables_num):
            # Initialize the new design variables
            target_solution = pre_solution.copy()
            next_target_solution = next_solution.copy()
            # change the value of design variables
            target_solution[target_idx] = next_target_solution[target_idx]
            # add the list called arguments sets
            arguments_sets.append([target_solution, pre_solution])
            # update the target design variables index
            target_idx += 1

        # calculate gradient
        diffs = []

        for idx, arg_set in enumerate(arguments_sets):
            next_args, pre_args = arg_set
            # calculate numerical differential
            diff = (self.objective_func.fitness(next_args) - self.objective_func.fitness(pre_args)) / self.diff_step

            diffs.append(diff)

        return diffs

    # steepest descent method
    def solve(self, init_solution):
        """
        conduct optimization of conjugate gradient method

        :param init_solution: initial design variables
        :return: current_solution, current_objective
        """
        # Initialize solution
        pre_solution = init_solution
        # count of calculation
        calc_count = 0

        while True:

            if calc_count == 100:
                print(' Local Minimum does nt exists ')
                exit()

            # calculate gradient of function , in another words, fprime
            grad_func = self.calc_gradient(pre_solution)
            # determine normalized direction vector
            direction_vector = -np.array(grad_func) / np.linalg.norm(np.array(grad_func))

            # Initialize variables for line search
            alpha = 1.0
            alpha_step = 0.01
            current_objective = 0.0
            current_objective_old = -np.inf

            # coefficient for armijo condition
            armijo_coef = 0.7

            # count of line search
            line_search_count = 0
            max_line_search_count = (alpha - 0) / alpha_step

            # Line Search
            # Standard for judging the optimal convergence is armijo condition. In other cases, wolfe condition exists.
            while True:
                # create armijo condition tareget
                armijo_target_func = self.objective_func.fitness(pre_solution) +\
                                     armijo_coef * alpha * np.dot(np.array(grad_func), np.array(direction_vector))

                # current solution
                current_solution = [pre_solution[idx] + alpha * direction_vector[idx] for idx in range(self.design_variables_num)]

                # calculate the current value of objectives
                current_objective = self.objective_func.fitness(current_solution)

                # check convergence
                if current_objective < armijo_target_func:

                    break

                if line_search_count == max_line_search_count:

                    break

                line_search_count += 1
                alpha *= 0.5

            # after conducting the part of line search, calculate gradient
            grad_norm = current_objective - self.objective_func.fitness(pre_solution)
            dir_norm = np.linalg.norm(np.array(current_solution) - np.array(pre_solution))
            grad = grad_norm / np.sqrt(dir_norm)

            print('grad_norm:', grad_norm)
            print('grad:', grad)

            if abs(grad) < 1.0e-6 or abs(grad_norm) < 1.0e-8:

                return current_solution, current_objective

            # replace pre solution into current solution
            pre_solution = current_solution

            calc_count += 1






def test_cg():

    design_variables_num = 4
    design_variable_class = GradientTestDesignVariable(design_variables_num)
    # establish the objective function class
    objective_func = GradientTestFunc(design_variable_class, design_variables_num)
    # build Optimization method class
    cgm = ConjugateGradientMethod(objective_func)

    init_solution = [2, 1, 1, 0]
    final_solution, final_objective = cgm.solve(init_solution)

    print('-' * 5 + ' Optimization Results ' + '-' * 5)
    print('optimal design variables:', final_solution)
    print('optimal value:', final_objective)
    print('-' * 60)

def test_gd():

    design_variables_num = 4
    design_variable_class = GradientTestDesignVariable(design_variables_num)
    # establish the objective function class
    objective_func = GradientTestFunc(design_variable_class, design_variables_num)
    # build Optimization method class
    gdm = GradientDescentMethod(objective_func)

    init_solution = [2, 1, 1, 0]
    final_solution, final_objective = gdm.solve(init_solution)

    print('-' * 5 + ' Optimization Results ' + '-' * 5)
    print('optimal design variables:', final_solution)
    print('optimal value:', final_objective)
    print('-' * 60)

def test_gd_ieg():
    design_variables_num = 4
    # baseline arguments
    baseline_aircraft_name = 'A320'
    baseline_aircraft_type = 'normal'
    baseline_engine_name = 'V2500'
    baseline_propulsion_type = 'turbofan'

    # data path
    aircraft_data_path = './DataBase/aircraft_test.json'
    engine_data_path = './DataBase/engine_test.json'
    mission_data_path = './Missions/cargo1.0_passenger1.0_.json'

    # construct baseline args
    baseline_args = [(baseline_aircraft_name, baseline_aircraft_type), (baseline_engine_name, baseline_propulsion_type),
                     (aircraft_data_path, engine_data_path)]

    # current arguments
    current_aircraft_name = 'A320'
    current_aircraft_type = 'normal'
    current_engine_name = 'V2500'
    current_propulsion_type = 'turbofan'

    # construct current args
    current_args = [(current_aircraft_name, current_aircraft_type), (current_engine_name, current_propulsion_type),
                    (aircraft_data_path, engine_data_path)]

    # off design point parameters
    off_altitude = 0
    off_mach = 0
    off_required_thrust = 133000  # [N]

    off_param_args = [off_altitude, off_mach, off_required_thrust]

    # constraint type : other option => 'COT', 'COBH', 'DFAN', 'LFAN'
    constraint_type = 'TIT'

    # build objective function class
    objective_func = IntegrationEnvGradient(baseline_args, current_args, off_param_args, mission_data_path,
                                            constraint_type, engine_mounting_positions=[0.2, 0.2], ld_calc_type='static-constraint')
    # build Optimization method class
    gdm = GradientDescentMethod(objective_func)

    # ToDo most important factor of making success for optimization is the selection of initial design point
    init_solution = [3, 25, 1.5, 1400]
    final_solution, final_objective = gdm.solve(init_solution)

    print('-' * 5 + ' Optimization Results ' + '-' * 5)
    print('optimal design variables:', final_solution)
    print('optimal value:', final_objective)
    print('-' * 60)



def test_scipy():
    design_variables_num = 4
    design_variable_class = GradientTestDesignVariable(design_variables_num)
    # establish the objective function class
    objective_func = GradientTestFunc(design_variable_class, design_variables_num)
    # build Optimization method class
    cgm = ConjugateGradientMethod(objective_func)


    def f(x, *args):

        return x[0] ** 3 + x[1] ** 3 + x[2] ** 3 - 3 * x[0] * x[1] - 3 * x[1] * x[2]

    def gradf(x, *args):

        design_variables_num, diff_step, f = args
        pre_solution = x.tolist()
        # no bounds
        next_solution = [pre_solution[idx] + diff_step for idx in range(design_variables_num)]

        # partial differential
        arguments_sets = []
        target_idx = 0

        for _ in range(design_variables_num):
            # Initialize the new design variables
            target_solution = pre_solution.copy()
            next_target_solution = next_solution.copy()
            # change the value of design variables
            target_solution[target_idx] = next_target_solution[target_idx]
            # add the list called arguments sets
            arguments_sets.append([target_solution, pre_solution])
            # update the target design variables index
            target_idx += 1

        # calculate gradient
        diffs = []

        for idx, arg_set in enumerate(arguments_sets):
            next_args, pre_args = arg_set
            # calculate numerical differential
            diff = (f(next_args) - f(pre_args)) / diff_step

            diffs.append(diff)

        return np.array(diffs)

    x0 = np.array([1, 1, 1, 1])
    args = (design_variables_num, 0.01, f)
    res1 = optimize.fmin_cg(f, x0, fprime=gradf, args=args)
    print(res1)




if __name__ == '__main__':
    # test_cg()
    # test_scipy()
    # test_gd()
    test_gd_ieg()






