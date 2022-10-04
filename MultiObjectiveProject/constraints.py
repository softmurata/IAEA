# In case of calculation at integration env class, include constraint index and value
class EnvConstraint(object):

    def __init__(self):

        # Turbine Inlet temperature => TIT
        self.tit = 1820  # [k]
        # Compressor Out Temperature => COT
        self.cot = 965  # [k]
        # Compressor Out Blade Height
        self.co_blade_h = 0.015  # [m]
        # Fan diameter
        self.fan_diam = 2.0  # [m]
        # Fan width
        self.fan_width = 3.0  # [m]

        # lp shaft revolve range
        self.rev_lp_min = 0.9
        self.rev_lp_max = 1.6

    def get_constraint_target(self, constraint_type):

        selected_constraint_target = None

        if constraint_type == 'TIT':

            selected_constraint_target = self.tit

        elif constraint_type == 'COT':

            selected_constraint_target = self.cot

        elif constraint_type == 'compressor_out_blade_height':

            selected_constraint_target = self.co_blade_h

        elif constraint_type == 'front_diameter':

            selected_constraint_target = self.fan_diam

        elif constraint_type == 'width_length_distributed_fan':

            selected_constraint_target = self.fan_width

        return selected_constraint_target


