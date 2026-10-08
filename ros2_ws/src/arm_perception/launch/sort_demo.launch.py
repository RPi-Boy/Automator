"""Whole demo in one command: simulated world + perception + planner + viewer window.

  ros2 launch arm_perception sort_demo.launch.py
  ros2 launch arm_perception sort_demo.launch.py model_path:=$AUTOMATOR_ROOT/models/perception_fp16.engine
  ros2 launch arm_perception sort_demo.launch.py inject_unknown:=true      # e-stop test
  ros2 launch arm_perception sort_demo.launch.py viewer:=false             # headless
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    root = os.environ.get('AUTOMATOR_ROOT', os.getcwd())
    args = [
        DeclareLaunchArgument('model_path', default_value=os.path.join(root, 'models', 'perception.onnx')),
        DeclareLaunchArgument('objects', default_value='4'),
        DeclareLaunchArgument('seed', default_value='0', description='scene layout; change for a different arrangement'),
        DeclareLaunchArgument('inject_unknown', default_value='false'),
        DeclareLaunchArgument('camera_hz', default_value='3.0'),
        DeclareLaunchArgument('gui', default_value='false', description="PyBullet's own 3-D window"),
        DeclareLaunchArgument('viewer', default_value='true', description='OpenCV demo window'),
        DeclareLaunchArgument('save_video', default_value='', description='write the viewer to this mp4'),
    ]
    return LaunchDescription(args + [
        Node(package='arm_perception', executable='sim_bridge_node', output='screen', parameters=[{
            'objects': LaunchConfiguration('objects'), 'seed': LaunchConfiguration('seed'), 'inject_unknown': LaunchConfiguration('inject_unknown'),
            'camera_hz': LaunchConfiguration('camera_hz'), 'gui': LaunchConfiguration('gui')}]),
        Node(package='arm_perception', executable='evidential_vision_node', output='screen',
             parameters=[{'model_path': LaunchConfiguration('model_path')}]),
        Node(package='arm_perception', executable='joint_trajectory_node', output='screen'),
        Node(package='arm_perception', executable='sim_viewer_node', output='screen',
             condition=IfCondition(LaunchConfiguration('viewer')),
             parameters=[{'save_video': LaunchConfiguration('save_video')}]),
    ])
