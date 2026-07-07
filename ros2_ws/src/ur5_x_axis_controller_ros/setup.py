from setuptools import find_packages, setup

PKG = "ur5_x_axis_controller_ros"

setup(
    name=PKG,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PKG}"]),
        (f"share/{PKG}", ["package.xml"]),
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
        "UR5e ROS 2 package. The CoppeliaSim controller/bridge nodes were archived to "
        "archive/coppelia/ros2/; the hardware pipeline node (ur5e_hardware_pipeline_node) "
        "was superseded by the plain-Python hardware/ lane rewrite and archived to "
        "archive/superseded/hardware_rtde_v1/ros2/ -- no ROS2 node is currently active "
        "in this package."
    ),
    license="TODO",
    tests_require=["pytest"],
    entry_points={"console_scripts": []},
)
