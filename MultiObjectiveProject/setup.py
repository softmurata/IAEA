
from distutils.core import setup,Extension
from Cython.Build import cythonize
import numpy


ext1=Extension("EngineComponent_utils",sources=["EngineComponent_utils.pyx"],include_dirs=[numpy.get_include()])
ext2=Extension("EngineComponentOff_utils", sources=["EngineComponentOff_utils.pyx"], include_dirs=[numpy.get_include()])
ext3=Extension("thermal_dp_cy", sources=["thermal_dp_cy.pyx"], include_dirs=[numpy.get_include()])
ext4=Extension("thermal_doff_cy", sources=["thermal_doff_cy.pyx"], include_dirs=[numpy.get_include()])
ext5=Extension("mission_tuning_cy", sources=["mission_tuning_cy.pyx"], include_dirs=[numpy.get_include()])
ext6=Extension("design_variable_cy", sources=["design_variable_cy.pyx"], include_dirs=[numpy.get_include()])
ext7=Extension("air_shape_utils", sources=["air_shape_utils.pyx"], include_dirs=[numpy.get_include()])
ext8=Extension("electric_weight_cy", sources=["electric_weight_cy.pyx"], include_dirs=[numpy.get_include()])
ext9=Extension("engine_weight_cy", sources=["engine_weight_cy.pyx"], include_dirs=[numpy.get_include()])
ext10=Extension("various_curve_utils", sources=["various_curve_utils.pyx"], include_dirs=[numpy.get_include()])
ext11=Extension("air_shape_cy", sources=["air_shape_cy.pyx"], include_dirs=[numpy.get_include()])


setup(name="off",ext_modules=cythonize([ext1, ext2, ext3, ext4, ext5, ext6, ext7, ext8, ext9, ext10, ext11]),include_dirs=[numpy.get_include()])