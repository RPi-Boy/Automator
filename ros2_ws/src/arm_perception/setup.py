from glob import glob

from setuptools import find_packages, setup

package_name = 'arm_perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Siddhant Kumar',
    maintainer_email='siddhant@teamavatar.io',
    description='ROS2 Perception and Evidential Vision Node for 4-DOF Manipulator',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'imx219_camera_node = arm_perception.imx219_camera_node:main',
            'evidential_vision_node = arm_perception.evidential_vision_node:main',
            'joint_trajectory_node = arm_perception.joint_trajectory_node:main',
            'sim_bridge_node = arm_perception.sim_bridge_node:main',
            'sim_viewer_node = arm_perception.sim_viewer_node:main',
        ],
    },
)
