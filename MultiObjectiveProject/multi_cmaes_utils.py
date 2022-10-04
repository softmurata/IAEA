import numpy as np
import math


class Individual(object):

    def __init__(self):
        self.rank = None
        self.objectives = None  # the values of objective indexes (type: list)
        self.normalized_objectives = None  # normalized values
        self.crowding_distance = None  # distance for crowdins
        self.design_vector = None  # the vector of design variables at evolutionary space
        self.therm_design_vector = None  # the vector of design variables at thermal space
        self.dominated_solutions = set()
        self.dominates = None
        self.domination_count = 0  # the number of points which dominates
        self.sigma = None  # variance
        self.cov = None  # covariance
        self.pEvol = None
        self.pSucc = None
        self.step = None


class Problem(object):

    def __init__(self, individual_num, objective_nums, objective_func_class, optimal_dir, objective_indexes=None):
        # the number of individuals which have design variables
        self.individual_num = individual_num
        # the number of objective indexes
        self.objective_nums = objective_nums
        # class for calculating the value of objective indexes
        self.objective_func_class = objective_func_class
        # the name's list of objectives
        self.objective_indexes = objective_indexes
        self.objectives = np.zeros((1, self.objective_nums))
        # the direction of optimization
        self.optimal_dir = optimal_dir

        if self.optimal_dir == 'downleft':
            self.optimal_dir_vec = [1.0 for _ in range(self.objective_nums)]
        else:
            self.optimal_dir_vec = [-1.0 for _ in range(self.objective_nums)]

        # calculate constant for this class
        self.calc_constants()

    # calculate global variables for Problem class
    def calc_constants(self):
        # target success probability
        self.ptarget = pow(5 + np.sqrt(0.5), -1)
        # step size damping parameters
        self.ddamping = 1.0 + self.individual_num * 0.5
        # success rate averaging parameter
        self.csuccrateparam = self.ptarget / (2.0 + self.ptarget)
        # cumulation time horizon parameter
        self.ccumultimespan = 2.0 / (2.0 + self.individual_num)
        # covariance matrix learning rate
        self.ccov = 2.0 / (pow(self.individual_num, 2) + 6)
        # p threshold
        self.pthreshold = 0.44

    def dominate(self, individual2, individual1):
        objective1_values = individual1.objectives
        objective2_values = individual2.objectives

        non_dominated = all(map(lambda f: f[0] <= f[1], zip(objective1_values, objective2_values)))
        dominates = any(map(lambda f: f[0] < f[1], zip(objective1_values, objective2_values)))

        return non_dominated and dominates

    def mutate(self, individual):
        while True:
            # set the required arguments
            mean = individual.design_vector
            cov = individual.sigma ** 2 * individual.cov
            # create new design vector according to the multivariate normal distribution
            new_design_vector = (np.random.multivariate_normal(mean, cov)) / np.linalg.norm(cov ** 2)

            # print('new design vector:', new_design_vector)

            # update individual class
            new_individual = Individual()
            new_individual.design_vector = new_design_vector
            if new_individual.cov is None:
                new_individual.cov = individual.cov
                new_individual.pEvol = individual.pEvol
                new_individual.pSucc = individual.pSucc
                new_individual.sigma = individual.sigma

            if self.objective_func_class.name != 'test':
                # generate thermal design variables
                si_design_variable_collect = [new_design_vector.tolist()]
                self.objective_func_class.design_variable.si_design_variable_collect = si_design_variable_collect
                self.objective_func_class.design_variable.generate_therm_design_variable_collect()
                new_individual.therm_design_vector = self.objective_func_class.design_variable.therm_design_variable_collect[0]

            meet_flag = self.calculate_objectives(new_individual)

            if meet_flag:
                break

        # whether new point dominates or not
        new_individual.dominates = self.dominate(individual, new_individual)
        # calculate step
        new_individual.step = (new_individual.design_vector - individual.design_vector) / individual.sigma

        return new_individual

    # update step size
    def update_step_size(self, new_individual):
        """

        :param new_individual: individual class after mutation
        :return: None
        """
        if new_individual.dominates:
            # if mutation is successful, the degree of pSucc is increasing
            new_individual.pSucc = (1.0 - self.csuccrateparam) * new_individual.pSucc + self.csuccrateparam
        else:
            new_individual.pSucc = (1.0 - self.csuccrateparam) * new_individual.pSucc

        # increase step size if success probability is bigger than target success probability(ptarget)
        new_individual.sigma = new_individual.sigma * np.exp((new_individual.pSucc - self.ptarget) / (self.ddamping * (1.0 - self.ptarget)))

    # update covariance
    def update_covariance(self, new_individual):
        """

        :param new_individual: individual class after mutation
        :return: None
        """
        if new_individual.pSucc < self.pthreshold:
            new_individual.pEvol = (1.0 - self.ccumultimespan) * new_individual.pEvol + np.sqrt(self.ccumultimespan * (2.0 - self.ccumultimespan)) * new_individual.step
            new_individual.cov = (1.0 - self.ccov) * new_individual.cov + self.ccov * (np.transpose(new_individual.pEvol) * new_individual.pEvol)
        else:
            new_individual.pEvol = (1.0 - self.ccumultimespan) * new_individual.pEvol
            new_individual.cov = (1.0 - self.ccov) * new_individual.cov + self.ccov * (np.transpose(new_individual.pEvol) * new_individual.pEvol + self.ccumultimespan * (2.0 - self.ccumultimespan) * new_individual.cov)

    def calculate_objectives(self, individual):
        individual.objectives = []
        individual.normalized_objectives = []

        # create function values
        # for test function
        if self.objective_func_class.name == 'test':
            objective_values = self.objective_func_class.fitness(individual.design_vector)

        else:
            objective_values_dict = self.objective_func_class.fitness(individual.therm_design_vector)

            # In case of not being able to find the answer
            if objective_values_dict is None:
                return False

            # create list of objective values
            objective_values = []

            for key, value in objective_values_dict.items():
                if key in self.objective_indexes:
                    objective_values.append(value)

        if objective_values is None:
            return False

        # construct the objectives for optimization
        for idx, obj in enumerate(objective_values):

            # the coefficient of optimal direction (-1 or 1)
            optimal_dif_coef = self.optimal_dir_vec[idx]

            obj *= optimal_dif_coef

            if self.objectives.shape[0] == 1:
                self.objectives[0, idx] = obj

            # calculate the mean and variance for normalization
            mean = self.objectives[:, idx].mean()
            var = self.objectives[:, idx].std()

            norm_obj = (obj - mean) / var

            individual.objectives.append(obj)
            individual.normalized_objectives.append(norm_obj)

        self.objectives = np.append(self.objectives, np.array([individual.objectives]), 0)

        return True


# utils for decomposing the covariance array
def tred2(n, V, d, e):
    """Symmetric Householder reduction to tridiagonal form.

    This is derived from the Algol procedures tred2 by Bowdler, Martin,
    Reinsch, and Wilkinson, Handbook for Auto. Comp., Vol.ii-Linear Algebra,
    and the corresponding Fortran subroutine in EISPACK.
    """
    for j in range(n):
        d[j] = V[n - 1][j]

    for i in range(n - 1, 0, -1):
        scale = 0.0
        h = 0.0

        for k in range(i):
            scale += abs(d[k])

        if scale == 0.0:
            e[i] = d[i - 1]

            for j in range(i):
                d[j] = V[i - 1][j]
                V[i][j] = V[j][i] = 0.0

        else:
            for k in range(i):
                d[k] /= scale
                h += d[k] ** 2

            f = d[i - 1]
            g = math.sqrt(h)

            if f > 0.0:
                g = -g

            e[i] = scale * g
            h -= f * g
            d[i - 1] = f - g

            for j in range(i):
                e[j] = 0.0

            for j in range(i):
                f = d[j]
                V[j][i] = f
                g = e[j] + V[j][j] * f

                for k in range(j + 1, i):
                    g += V[k][j] * d[k]
                    e[k] += V[k][j] * f

                e[j] = g

            f = 0.0

            for j in range(i):
                e[j] /= h
                f += e[j] * d[j]

            hh = f / (2 * h)

            for j in range(i):
                e[j] -= hh * d[j]

            for j in range(i):
                f = d[j]
                g = e[j]

                for k in range(j, i):
                    V[k][j] -= f * e[k] + g * d[k]

                d[j] = V[i - 1][j]
                V[i][j] = 0.0

        d[i] = h

    for i in range(n - 1):
        V[n - 1][i] = V[i][i]
        V[i][i] = 1.0
        h = d[i + 1]

        if h != 0.0:
            for k in range(i + 1):
                d[k] = V[k][i + 1] / h

            for j in range(i + 1):
                g = 0.0

                for k in range(i + 1):
                    g += V[k][i + 1] * V[k][j]

                for k in range(i + 1):
                    V[k][j] -= g * d[k]

        for k in range(i + 1):
            V[k][i + 1] = 0.0

    for j in range(n):
        d[j] = V[n - 1][j]
        V[n - 1][j] = 0.0

    V[n - 1][n - 1] = 1.0
    e[0] = 0.0


def tql2(n, d, e, V):
    """Symmetric tridiagonal QL algorithm.

    This is derived from the Algol procedures tql2, by Bowdler, Martin,
    Reinsch, and Wilkinson, Handbook for Auto. Comp., Vol.ii-Linear Algebra,
    and the corresponding Fortran subroutine in EISPACK.
    """
    for i in range(1, n):
        e[i - 1] = e[i]

    e[n - 1] = 0.0

    f = 0.0
    tst1 = 0.0
    eps = math.pow(2.0, -52.0)

    for l in range(n):
        tst1 = max(tst1, abs(d[l]) + abs(e[l]))
        m = 1

        while m < n:
            if abs(e[m]) <= eps * tst1:
                break
            m += 1

        if m > l:
            iter = 0

            while True:
                iter += 1
                g = d[l]
                p = (d[l + 1] - g) / (2.0 * e[l])
                r = hypot(p, 1.0)

                if p < 0:
                    r = -r

                d[l] = e[l] / (p + r)
                d[l + 1] = e[l] * (p + r)
                dl1 = d[l + 1]
                h = g - d[l]

                for i in range(l + 2, n):
                    d[i] -= h

                f += h
                p = d[m]
                c = 1.0
                c2 = c
                c3 = c
                el1 = e[l + 1]
                s = 0.0
                s2 = 0.0

                for i in range(m - 1, l - 1, -1):
                    c3 = c2
                    c2 = c
                    s2 = s
                    g = c * e[i]
                    h = c * p
                    r = hypot(p, e[i])
                    e[i + 1] = s * r
                    s = e[i] / r
                    c = p / r
                    p = c * d[i] - s * g
                    d[i + 1] = h + s * (c * g + s * d[i])

                    for k in range(n):
                        h = V[k][i + 1]
                        V[k][i + 1] = s * V[k][i] + c * h
                        V[k][i] = c * V[k][i] - s * h

                p = -s * s2 * c3 * el1 * e[l] / dl1
                e[l] = s * p
                d[l] = c * p

                if abs(e[l]) <= eps * tst1:
                    break

        d[l] = d[l] + f
        e[l] = 0.0

    for i in range(n - 1):
        k = i
        p = d[i]

        for j in range(i + 1, n):
            if d[j] < p:
                k = j
                p = d[j]

        if k != i:
            d[k] = d[i]
            d[i] = p

            for j in range(n):
                p = V[j][i]
                V[j][i] = V[j][k]
                V[j][k] = p


# utils for non dominated sort
def non_dominated_cmp(x, y):

    if x.rank == y.rank:
        if -x.crowding_distance < -y.crowding_distance:
            return -1
        elif -x.crowding_distance > -y.crowding_distance:
            return 1
        else:
            return 0

    else:
        if x.rank < y.rank:
            return -1
        elif x.rank > y.rank:
            return 1
        else:
            return 0

# ToDo : In order to implement following function, we have to confirm whether archive class is needed or not
# ToDo : if archive class is needed, we have to consider how to establish the archive class.
def non_dominated_sort(population):

    rank = 0
    while len(population) > 0:

        rank += 1

def crowding_distance(population):
    pass

# class Archive(Dominance):
# class ParetoDominance():




