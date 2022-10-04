from integration_env_explore import IntegrationEnvExplore


# class for converting the values which have the thermal and aerodynamic features into those which have optimization
class IntegrationEnvGradient(IntegrationEnvExplore):
    """



    """

    def __init__(self, baseline_args, current_args, off_param_args, mission_data_path, constraint_type,
                 engine_mounting_positions=[0.2, 0.2], ld_calc_type='static-constant'):

        super().__init__(baseline_args, current_args, off_param_args, mission_data_path, constraint_type,
                         engine_mounting_positions, ld_calc_type)
        # name which shows variables of class
        self.name = 'gradient'
        # the count of calling fitness function

    def fitness(self, solution):
        """

        :param solution:
        :return: objectives
        """
        # solution shape is design variables only
        self.design_variable.si_design_variable_collect = [solution]
        self.design_variable.generate_therm_design_variable_collect()
        thermal_design_variables = self.design_variable.therm_design_variable_collect[0]

        # results of objectives
        calc_result_list = self.run_meet_constraints(thermal_design_variables)

        # extract the value of fuelburn
        objectives = calc_result_list[0]

        return objectives



