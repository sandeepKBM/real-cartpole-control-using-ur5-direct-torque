from pathlib import Path

from setuptools import find_packages, setup

PKG = "ur5_x_axis_controller_ros"

setup(
    name=PKG,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PKG}"]),
        (f"share/{PKG}", ["package.xml"]),
        (f"share/{PKG}/launch", ["launch/run_controller.launch.py"]),
        (f"share/{PKG}/launch", ["launch/run_ur5e_hardware_pipeline.launch.py"]),
        (
            f"share/{PKG}/config",
            [
                "config/controller.yaml",
                "config/controller_coppelia_legacy_xz_transport.yaml",
            ],
        ),
    ],
    install_requires=[
        "setuptools",
        "numpy",
        "pyyaml",
        "coppeliasim-zmqremoteapi-client",
    ],
    zip_safe=True,
    maintainer="ss5772",
    maintainer_email="ss5772@users.noreply",
    description=(
        "UR5 X-axis torque controller (Cartesian-X PD + J^T). "
        "Wraps controller_core and talks to CoppeliaSim via ZMQ Remote API."
    ),
    license="TODO",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            f"controller_node = {PKG}.controller_node:main",
            f"coppeliasim_bridge_node = {PKG}.coppeliasim_bridge_node:main",
            f"ur5e_hardware_pipeline_node = {PKG}.ur5e_hardware_pipeline_node:main",
        ],
    },
)
