import numpy as np
cimport numpy as np

cdef class DistributedFanWeight(object):
    cdef np.ndarray cref
    cdef np.ndarray cref_e
    cdef np.ndarray dref
    cdef np.ndarray qref
    cdef np.ndarray qref_e
    cdef np.ndarray coff
    cdef np.ndarray coff_e
    cdef np.ndarray doff
    cdef np.ndarray qoff
    cdef np.ndarray qoff_e
    cdef double fscl
    cdef double Kf
    cdef double AR
    cdef double sigma_t
    cdef double sigma_t_ref
    cdef double u_tip_ref
    cdef double density
    cdef double duct_length
    cdef double Nfan
    cdef double fan_tip_hub_ratio
    cdef double aspect_ratio_rotor
    cdef double aspect_ratio_stator
    cdef double distributed_fan_weight
    cdef double distributed_fan_duct_weight
    cdef double distributed_fan_length
    cdef double distributed_fan_width_length
    cdef list fan_in_diameter
    cdef list fan_out_diameter


cdef class ElectricEquipWeight(object):
    cdef np.ndarray cref
    cdef np.ndarray cref_e
    cdef np.ndarray dref
    cdef np.ndarray qref
    cdef np.ndarray qref_e
    cdef np.ndarray coff
    cdef np.ndarray coff_e
    cdef np.ndarray doff
    cdef np.ndarray qoff
    cdef np.ndarray qoff_e
    cdef double electric_component_density
    cdef double gene_power
    cdef battery_types
    cdef double k_cw
    cdef double electric_equip_weight
    cdef double generator_weight
    cdef double battery_weight



