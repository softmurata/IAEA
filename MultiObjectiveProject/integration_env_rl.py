import numpy as np
from integration_env_explore import IntegrationEnvExplore


class IntegrationEnvRL(IntegrationEnvExplore):
    """
    Attributes
    ------------
    name: str

    """

    def __init__(self, baseline_args, current_args, off_param_args, mission_data_path, constraint_type, engine_mounting_positions, ld_calc_type):
        super().__init__(baseline_args, current_args, off_param_args, mission_data_path, constraint_type, engine_mounting_positions, ld_calc_type)

        self.name = 'rl'

        # create design space for reinforcement learning
        self.rl_design_space = []
        for dv_range in self.design_variable.design_space_dict.vals():
            self.rl_design_space.append(dv_range)

    def step(self, rl_design_variables, other_requirements):
        """

        :param rl_design_variables: list
                                    not thermal design variables, but same as them in swarm intelligence
        :param other_requirements: list
                                   [eval_successions, reward_successions, eval_function, previous_eval_function]
        :return: state, reward, terminal, info, other_requirements


        Attributes
        -------------
        state: list

        reward: float

        terminal: boolean

        info: dict

        other_requirements: list

        eval_successions: int

        reward_successions: int

        eval_func: float

        previous_eval_func: float

        """
        # replace the design variables into state, which is thought of as reinforcement learning's space
        state = rl_design_variables
        # convert the reinforcement learning's design variables into thermal design variables
        self.design_variable.si_design_variable_collect = [rl_design_variables]
        self.design_variable.generate_therm_design_variable_collect()
        thermal_design_variables = self.design_variable.therm_design_variable_collect[0]

        calc_results_list = self.run_meet_constraints(thermal_design_variables)

        # make reward and terminal
        reward, terminal, other_requirements = self.get_reward_and_terminal(calc_results_list, state, other_requirements)
        # create info dictionary
        info = self.get_info(calc_results_list, reward, terminal, state)

        return state, reward, terminal, info, other_requirements

    def get_reward_and_terminal(self, calc_results_list, state, other_requirements):
        """
        determine reward and the moment of terminating this game comparing with constraint conditions and update other parameters on the reinforcement learning process

        :param calc_results_list: list
        :param state: list
        :param other_requirements: list
        :return: reward, terminal, other_requirements
        """
        # In this case, single objective is fuel burn
        single_objective = calc_results_list[0]
        eval_successions, reward_successions, eval_func, previous_eval_func = other_requirements
        # default
        reward = 0
        terminal = False

        if single_objective is None:
            reward = -1
            terminal = True
            eval_successions = 0
            reward_successions = 0
            previous_eval_func = 1
        else:

            eval_func = single_objective / self.init_mission_class.fuel_weight

            # Case1 the eval function value is lower than target one
            if eval_func <= 1.0:
                eval_successions += 1
            else:
                eval_successions += 1

            # Case2 the current eval function's value is larger than previous one
            if eval_func > previous_eval_func:
                reward = -1
                reward_successions = 0
                terminal = False

            elif eval_func <= previous_eval_func:
                reward = 2
                reward_successions += 1
                terminal = False

        # Case 3 the design variables is beyond the range of design space
        for idx, dv_range in enumerate(self.rl_design_space):
            dv_flag = False
            if state[idx] < dv_range[0] or state[idx] > dv_range[1]:
                reward = -1
                dv_flag = True
            else:
                reward = 0
                dv_flag = False

            if dv_flag is True:
                terminal = True
                break

        other_requirements = [eval_successions, reward_successions, eval_func, previous_eval_func]

        return reward, terminal, other_requirements

    def get_info(self, calc_results_list, reward, terminal, state):
        """
        compose of entire information in one step of reinforcement learning process

        :param calc_results_list: list
        :param reward: float
        :param terminal: boolean
        :param state: list
        :return: info
        """

        info = {}
        # add objective indexes
        for name, idx in self.results_list_index_dict.items():
            info[name] = calc_results_list[idx]

        # add state, reward, terminal
        info['state'] = state
        info['reward'] = reward
        info['terminal'] = terminal

        return info




