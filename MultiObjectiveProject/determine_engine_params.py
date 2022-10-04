import numpy as np
import json
from engine_weight import EngineWeight
from thermal_dp import CalcDesignPoint
from thermal_doff import CalcDesignOffPoint

# Determine engine design variables
# Survey the public values of design variables you want to determine from website
# ex) https://www.cfmaeroengines.com/engines/cfm56/
# In most cases, the baseline engine type is turbofan engine
# so you should find the following values of design variables
# BPR, OPR, FPR, TIT
# And tuning target values are following this
# SFC (Specific Fuel Consumption), Fan Diameter, Thrust(@ ground or Takeoff), airflow rate(ks/s), engine weight[kg]
# if you can find some values of design variables, you have to fix this value while tuning
# And You have to change other design variables you cannot find so that tuning target values are suitable

class EngineTuning(object):

    def __init__(self, aircraft_name, engine_name, aircraft_type, propulsion_type, off_param_args, data_base_args):

        self.aircraft_name = aircraft_name
        self.engine_name = engine_name
        self.aircraft_type = aircraft_type
        self.propulsion_type = propulsion_type
        self.data_base_args = data_base_args
        self.off_param_args = off_param_args
        self.aircraft_data_path, self.engine_data_path, self.mission_data_path = data_base_args

        # data file
        f = open(self.aircraft_data_path, 'r')
        self.aircraft_data_file = json.load(f)[self.aircraft_name]

        f = open(self.engine_data_path, 'r')
        self.engine_data_file = json.load(f)[self.engine_name]

    def judge_restrict(self, thermal_design_variables):
        # lp shaft revolve rate
        # change rev lp and find thrust value
        rev_lp = 0.9
        rev_lp_step = 0.01

        # minimum residual
        restarget = 0.0
        restargetold = 0.0

        # residual target names
        residual_target_names = ['Thrust', 'sfc_ground']

        # build calc off design point class
        self.calc_off_design_point_class = CalcDesignOffPoint(self.aircraft_name, self.engine_name, self.aircraft_type,
                                                              self.propulsion_type, thermal_design_variables,
                                                              self.off_param_args, self.data_base_args)

        while True:

            rev_args = [rev_lp, 1.0]

            self.calc_off_design_point_class.run_off(rev_args)

            self.calc_off_design_point_class.objective_func_doff()

            # calculate residual
            resthrust = 1.0 - self.calc_off_design_point_class.thrust_off / \
                       self.calc_off_design_point_class.required_thrust_ground
            ressfcoff = 1.0 - self.calc_off_design_point_class.sfc_off / self.engine_data_file['sfc_ground']

            residuals = [abs(resthrust), abs(ressfcoff)]

            for name, val in zip(residual_target_names, residuals):
                print('objective_index:{0}, residual_value:{1}'.format(name, val))

            restarget = min(residuals)
            # check convergence
            if abs(restarget) < 1.0e-7:
                break

            if restarget * restargetold < 0.0:
                rev_lp_step *= 0.5

            restargetold = restarget

            # update lp shaft revolve
            rev_lp += np.sign(restarget) * rev_lp_step

    def calc_performance(self, thermal_design_variables):

        # judge thrust convergence
        self.judge_restrict(thermal_design_variables)


    def determine_engine_stage_number(self):

        # build engine weight class
        engine_weight_class = EngineWeight(self.aircraft_name, self.engine_name, self.aircraft_type,
                                           self.propulsion_type, self.calc_off_design_point_class,
                                           self.engine_data_path)


# test for Quasi newton method
# ToDo we have to implement the quasi newton method
class QuasiNewton(object):

    def __init__(self, design_variables_num, solution_range, objective_func):

        self.design_variables_num = design_variables_num
        self.solution_range = solution_range
        self.diff_step = 0.01  # for numerical differential
        self.objective_func = objective_func

    def calc_gradient(self, args):
        next_args = [args[idx] + self.diff_step for idx in range(2)]

        # calculate partial differentiation
        arguments_sets = []
        target_idx = 0
        for _ in range(2):
            target_solution = args.copy()
            next_target_solution = next_args.copy()
            target_solution[target_idx] = next_target_solution[target_idx]
            arguments_sets.append([target_solution, args])
            target_idx += 1

        # Gradient
        diffs = []

        for idx, arg_set in enumerate(arguments_sets):
            next_args, pre_args = arg_set
            diff = (self.objective_func(next_args) - self.objective_func(pre_args)) / self.diff_step

            diffs.append(diff)

        return diffs

    def calc_heassian_arr(self, current_solution, alpha, direction):
        """

        :param current_solution: list:[current optimize point]
        :param alpha: float
        :param direction: list (finding direction)
        :return:
        """

        pass

    def solve(self, init_solution):

        # calculate gradient of objective functions and hessian array
        # Line search
        # update solution by gradient
        Hk = np.identity(self.design_variables_num)
        yk = [0.1, 0.1]


# conjugate gradient
class ConjugateGradient(object):

    def __init__(self, design_variables_num, diff_step, solution_range, objective_func):

        self.design_variables_num = design_variables_num
        # for numerical differential
        self.diff_step = diff_step
        # range of exploration of solution
        self.solution_range = solution_range
        # objective function
        self.objective_func = objective_func

    def calc_gradient(self, pre_solution):
        next_solution = [pre_solution[idx] + self.diff_step for idx in range(self.design_variables_num)]

        # partial differential
        # calculate partial differentiation
        arguments_sets = []
        target_idx = 0
        for _ in range(self.design_variables_num):
            target_solution = pre_solution.copy()
            next_target_solution = next_solution.copy()
            target_solution[target_idx] = next_target_solution[target_idx]
            arguments_sets.append([target_solution, pre_solution])
            target_idx += 1

        # Gradient
        diffs = []

        for idx, arg_set in enumerate(arguments_sets):
            next_args, pre_args = arg_set
            diff = (self.objective_func(next_args) - self.objective_func(pre_args)) / self.diff_step

            diffs.append(diff)

        return diffs

    def solve(self, init_solution):
        # Initialize previous solution
        pre_solution = init_solution
        # Initialize search direction
        grad_func = self.calc_gradient(pre_solution)
        direction = [-g for g in grad_func]
        # armijo coefficient
        armijo_coef = 0.8

        calc_count = 0

        while True:

            if calc_count == 30:
                exit()

            # Line Search
            alpha = 1.0
            alpha_step = 0.01
            max_alpha_step = alpha / alpha_step
            alpha_count = 0

            pre_grad_func = self.calc_gradient(pre_solution)

            while True:
                # convergence target
                armijo_target = self.objective_func(pre_solution) + armijo_coef * alpha * \
                                np.array(pre_grad_func).dot(np.array(direction))
                current_solution = [pre_solution[idx] + alpha * direction[idx] for idx in range(self.design_variables_num)]

                current_obj = self.objective_func(current_solution)

                if current_obj < armijo_target:

                    # print('optimal line search')
                    print(alpha)
                    break

                if alpha_count == max_alpha_step:
                    exit()

                # update alpha step
                alpha -= alpha_step
                alpha_count += 1

            # current exploration coefficients
            current_grad_func = self.calc_gradient(current_solution)
            beta = np.linalg.norm(np.array(current_grad_func)) / np.linalg.norm(np.array(pre_grad_func))

            # convergence check
            dir_norm = np.linalg.norm(np.array(direction)) * alpha
            resobj = current_obj - self.objective_func(pre_solution)
            print('direction norm:', dir_norm)
            print('residual objective function:', resobj)

            if abs(resobj) < 1.0e-6:

                print('Optimal')

                return current_solution, current_obj

            # update direction
            direction = [-current_grad_func[idx] + beta * direction[idx] for idx in range(self.design_variables_num)]

            calc_count += 1

            pre_solution = current_solution





# test for gradient method
# cannot solve the solution in case of
class Gradient(object):

    def __init__(self, design_variables_num, vec_step, solution_range, objective_func):

        self.design_variables_num = design_variables_num
        # Difference of vector
        self.vec_step = vec_step
        self.vector_step = [vec_step] * self.design_variables_num
        self.objective_func = objective_func
        # value range of solutions
        self.solution_range = solution_range

    def calc_gradient(self, pre_solution):
        next_solution = [pre_solution[idx] + self.vector_step[idx] for idx in range(self.design_variables_num)]

        # print(pre_solution)
        # print(next_solution)

        # calculate partial differentiation
        arguments_sets = []
        target_idx = 0
        for _ in range(self.design_variables_num):
            target_solution = pre_solution.copy()
            next_target_solution = next_solution.copy()
            target_solution[target_idx] = next_target_solution[target_idx]
            arguments_sets.append([target_solution, pre_solution])
            target_idx += 1


        # Gradient
        diffs = []

        for idx, arg_set in enumerate(arguments_sets):
            next_args, pre_args = arg_set
            diff = (self.objective_func(next_args) - self.objective_func(pre_args)) / self.vec_step

            diffs.append(diff)

        return diffs

    def steepest_descent(self, init_solution):

        # Initialize solution
        pre_solution = init_solution

        calc_count = 0

        while True:

            if calc_count == 100:
                print(' Local Minimum does not exists ')
                exit()

            # calculate gradient
            grad_func = self.calc_gradient(pre_solution)
            # print('function prime:', grad_func)

            # direction
            direct_vector = - np.array(grad_func) / np.linalg.norm(np.array(grad_func))
            # print('direction vector:', direct_vector)
            # print('now solution:', pre_solution)

            alpha = 1.0
            alpha_step = 0.01

            current_obj = 0.0
            current_objold = -np.inf

            alpha_count = 0
            max_alpha_count = alpha / alpha_step

            # LineSearch

            while True:

                armijo_target_func = self.objective_func(pre_solution) + \
                                     0.7 * alpha * np.array(grad_func).dot(np.array(direct_vector))

                current_solution = [pre_solution[idx] + alpha * direct_vector[idx] for idx in range(self.design_variables_num)]

                current_obj = self.objective_func(current_solution)

                # print(alpha, current_obj, armijo_target_func)

                if current_obj < armijo_target_func:

                    # print('optimal solution')

                    break

                if alpha_count == max_alpha_count:

                    break

                alpha_count += 1

                # alpha -= alpha_step
                alpha *= 0.5

            # compute gradient
            pre_obj = self.objective_func(pre_solution)
            grad_norm = current_obj - pre_obj
            dir_norm = np.linalg.norm(np.array(current_solution) - np.array(pre_solution))
            grad = grad_norm / np.sqrt(dir_norm)

            print('grad_norm:', grad_norm)
            print('grad:', grad)

            if abs(grad) < 1.0e-6:

                return current_solution, current_obj

            pre_solution = current_solution

            calc_count += 1


# test for gradient method
def test_grad():
    # objective function
    """
    # 2 dimension
    def f(args):

        return args[0]**2 - 2 * args[0] * args[1] + 3 * args[1]**2
    """


    # 3 dimension
    def f(args):

        return args[0] ** 4 + args[1] ** 3 + args[2] ** 2 + args[3] ** 3 - 2 * args[0] ** 2 * args[1] - 4 * args[0] * args[3]

    def f2(args):

        return args[0] ** 3 + args[1] ** 3 + args[2] ** 3 - 3 * args[0] * args[1] - 3 * args[1] * args[2]

    def f3(args):

        return args[0] ** 2 - 2 * (args[0] + args[1] + args[2]) + args[1] ** 2 + args[2] ** 2

    """
    # draw graph
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    x = np.arange(-5.0, 5.0, 0.1)
    y = np.arange(-5.0, 5.0, 0.1)

    X, Y = np.meshgrid(x, y)

    L = f3([X, Y, X])

    fig = plt.figure()

    ax = Axes3D(fig)

    ax.plot_surface(X, Y, L)

    plt.show()
    """


    design_variables_num = 4
    vec_step = 0.001
    solution_range = [[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]]

    # Normal gradient method
    gd = Gradient(design_variables_num, vec_step, solution_range, objective_func=f2)

    # init solution
    init_solution = [0.0, 0.8, 0.8, 0.6]
    ans_solution, ans_objectives = gd.steepest_descent(init_solution)
    print('Final results')
    print(ans_solution, ans_objectives)

    # Conjugate gradient method
    # cg = ConjugateGradient(design_variables_num, vec_step, solution_range, objective_func=f3)
    # ans_solution, ans_objectives = cg.solve(init_solution)
    # print('Final results')
    # print(ans_solution, ans_objectives)


if __name__ == '__main__':
    test_grad()
    exit()

    def f(args):

        return args[0] ** 3 + args[1] ** 3 - 3 * args[0] * args[1]

    def grad_f(args, objective_func):
        diff_step = 0.01
        next_args = [args[idx] + diff_step for idx in range(2)]

        # calculate partial differentiation
        arguments_sets = []
        target_idx = 0
        for _ in range(2):
            target_solution = args.copy()
            next_target_solution = next_args.copy()
            target_solution[target_idx] = next_target_solution[target_idx]
            arguments_sets.append([target_solution, args])
            target_idx += 1

        # Gradient
        diffs = []

        for idx, arg_set in enumerate(arguments_sets):
            next_args, pre_args = arg_set
            diff = (objective_func(next_args) - objective_func(pre_args)) / diff_step

            diffs.append(diff)

        return diffs

    Hk = np.identity(2)
    # Initialize solution
    init_solution = [0.5, 0.5]
    grad_func = grad_f(init_solution, f)
    # direction
    direction = -np.dot(Hk, np.array(grad_func))
    print('direction vector:', direction)

    # Line Search
    alpha = 1.0
    alpha_step = 0.01
    max_alpha_step = int(alpha / alpha_step)

    # optimization condition
    armijo_coef = 0.8

    # Initialize pre_solution
    pre_solution = init_solution

    for _ in range(max_alpha_step):
        current_solution = [pre_solution[idx] + alpha * direction[idx] for idx in range(2)]
        pre_grad_func = grad_f(pre_solution, f)
        armijo_target = f(pre_solution) + armijo_coef * alpha * np.dot(np.array(pre_grad_func), np.array(direction))

        current_obj = f(current_solution)

        if current_obj < armijo_target:

            break

        # update alpha
        alpha -= alpha_step

    # calculate sk yk
    sk = alpha * np.array(direction)
    current_grad_func = grad_f(current_solution, f)
    yk = [current_grad_func[idx] - pre_grad_func[idx] for idx in range(2)]
    yk = np.array(yk)
    I = np.identity(2)

    print(Hk.dot(sk))
    print(yk)

    Hk_next = (I - (sk.dot(yk.T))/(sk.T.dot(yk))) * Hk * (I - (yk.dot(sk))/(sk.T.dot(yk))) + sk.dot(sk.T)/(sk.T.dot(yk))
    print(Hk_next)

    grad_func = grad_f(current_solution, f)
    direction = -np.dot(Hk_next, np.array(grad_func))

    print('next solution:', current_solution)
    print('next direction vector:', direction)

