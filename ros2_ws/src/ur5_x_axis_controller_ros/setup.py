from setuptools import find_packages, setup

PKG = "ur5_x_axis_controller_ros"

setup(
    name=PKG,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PKG}"]),
        (f"share/{PKG}", ["package.xml"]),
        (f"share/{PKG}/launch", ["launch/run_ur5e_hardware_pipeline.launch.py"]),
    ],
    install_requires=[
        "setuptools",
        "numpy",
        "pyyaml",
    ],
    zip_safe=True,
    maintainer="ss5772",
    maintainer_email="ss5772@users.noreply",
    description=(
        "UR5e hardware pipeline ROS 2 node. "
        "The CoppeliaSim controller/bridge nodes were archived to archive/coppelia/ros2/."
    ),
    license="TODO",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            f"ur5e_hardware_pipeline_node = {PKG}.ur5e_hardware_pipeline_node:main",
        ],
    },
)
