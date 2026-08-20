import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

ext_modules = [
    Extension(
        "super_ik_tsinghua_1dqp._super_ik_tsinghua_1dqp",
        [
            "python/bindings.cpp",
            "src/solver.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
            "include",
            "/usr/include/eigen3",
        ],
        extra_compile_args=["-O3", "-std=c++17"],
        language="c++",
    ),
]

setup(
    name="super-ik-tsinghua-1dqp",
    version="0.1.0",
    package_dir={"": "python"},
    packages=["super_ik_tsinghua_1dqp"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
