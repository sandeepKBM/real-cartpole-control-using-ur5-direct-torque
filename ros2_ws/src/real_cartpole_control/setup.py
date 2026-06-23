from setuptools import find_packages, setup


package_name = "real_cartpole_control"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", ["config/origin_hold.yaml"]),
        (f"share/{package_name}/launch", ["launch/origin_hold.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ss5772",
    maintainer_email="ss5772@example.com",
    description="Control-first ROS package for the real UR5e integration path.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "origin_hold_controller = real_cartpole_control.origin_hold_controller:main",
        ],
    },
)
