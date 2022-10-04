import numpy as np
from deap import benchmarks


# Design variable class
class ObjectTestDesignVariable(object):

    def __init__(self, design_variables_num):
        # the number of design variables
        self.design_variables_num = design_variables_num
        # all bounds
        self.bounds = [[0, 0] for _ in range(self.design_variables_num)]

    def set_design_variable_bounds(self, bounds):

        self.bounds = bounds

    def norm(self, design_variables):

        new_design_variables = []

        for idx, dv in enumerate(design_variables):
            target_bound = self.bounds[idx]
            norm_dv = (dv - target_bound[0]) / (target_bound[1] - target_bound[0])
            new_design_variables.append(norm_dv)

        return new_design_variables

    def norm_for_individual(self, design_variables_collect):
        # the contents of design variables collect are individual classes
        for individual in design_variables_collect:
            target_design_variable = individual.design_vector
            norm_target_design_variable = self.norm(target_design_variable)
            # replace the normalized design vector
            individual.normalized_design_vector = norm_target_design_variable

        return design_variables_collect

    def norm_for_design_variables(self, design_variables_collect):

        new_design_variables_collect = []

        for target_design_variables in design_variables_collect:
            norm_target_design_variable = self.norm(target_design_variables)
            new_design_variables_collect.append(norm_target_design_variable)

        return new_design_variables_collect


class ObjectTestFunc(object):
    def __init__(self, objective_nums, design_variables_num, bounds):
        self.objective_nums = objective_nums
        self.design_variables_num = design_variables_num
        self.name = 'test'
        self.design_variable = ObjectTestDesignVariable(design_variables_num)
        # set the space of design variables
        self.design_variable.set_design_variable_bounds(bounds)

    def fitness(self, x, func_name='dtlz2'):

        function_class = Function()

        if func_name == 'dtlz1':

            ans = benchmarks.dtlz1(x, self.objective_nums)

            return ans

        elif func_name == 'dtlz2':

            return benchmarks.dtlz2(x, self.objective_nums)

        elif func_name == 'mmop':

            function_class.select(x, func_name)

            return function_class.objectives

        return None


class Function(object):

    def __init__(self):
        self.x = None
        self.objectives = None

    def select(self, x, name):
        if name == 'mmop':

            self.mmop(x)

    def mmop(self, x):

        f1 = x[0]
        f2 = 0
        g = 1.0 + 10 * (len(x) - 1)

        for xi in x[1:]:
            g += (xi ** 2 - 10 * np.cos(2.0 * np.pi * xi))

        h = 0

        if f1 <= g:
            h = 1.0 - (f1 / g) ** 0.5

        f2 = g * h

        objectives = [f1, f2]

        self.x = x
        self.objectives = objectives



def test():

    objective_nums = 3
    design_variables_num = 7
    bounds = [[0, 1], [-10, 10], [-10, 10], [-10, 10], [-10, 10], [-10, 10], [-10, 10]]

    otf = ObjectTestFunc(objective_nums, design_variables_num, bounds)

    dv_collect = []

    for _ in range(1000):
        x = [np.random.rand() for _ in range(design_variables_num)]
        dv_collect.append(x)

    objs = []
    func_name = 'dtlz2'

    for dv in dv_collect:

        objs.append(otf.fitness(dv, func_name))

    dvs = np.array(dv_collect)
    objs = np.array(objs)

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    if objective_nums == 3:
        fig = plt.figure(figsize=(12, 8))
        ax = Axes3D(fig)

        ax.scatter(objs[:, 0], objs[:, 1], objs[:, 2])

        ax.set_xlabel('f1')
        ax.set_ylabel('f2')
        ax.set_zlabel('f3')

        plt.show()

    elif objective_nums == 2:
        plt.figure(figsize=(10, 6))
        plt.scatter(objs[:, 0], objs[:, 1])
        plt.show()


# meaningless code
def test_meaningless():
    class Attacker(object):

        def __init__(self, name, attack, defense, hp):
            self.name = name
            self.attack = attack
            self.defense = defense
            self.hp = hp
            self.attack_nums = None
            self.attack_damages = None

        def set_attack_names(self, attack_damages):
            self.attack_damages = attack_damages
            self.attack_nums = len(attack_damages)

        def attack(self):

            return np.random.randint(self.attack_nums)

        def damage(self, attacker_name):
            self.hp -= (attacker_name.attack - self.defense)

        def judge_dead(self):
            if self.hp <= 0.0:
                print('dead')

    luffy = Attacker(name='Luffy', attack=100, defense=90, hp=200)
    luffy_attack_damages = [1.1, 1.5, 1.7, 1.9, 3.5]
    luffy.set_attack_names(luffy_attack_damages)

    zoro = Attacker(name='Zoro', attack=80, defense=80, hp=180)
    zoro_attack_damages = [1.3, 1.8, 2.0, 4.0]
    zoro.set_attack_names(zoro_attack_damages)

    attackers_collection = [luffy, zoro]

    for attacker in attackers_collection:
        print(attacker.name, ':', attacker.attack_damages)

    for attacker in attackers_collection:

        new_attack_damages = []
        for attack_coef in attacker.attack_damages:
            attack_coef *= 1.1
            new_attack_damages.append(attack_coef)

        attacker.attack_damages = new_attack_damages

    for attacker in attackers_collection:
        print(attacker.name, ':', attacker.attack_damages)


if __name__ == '__main__':

    test()
    # test_meaningless()


