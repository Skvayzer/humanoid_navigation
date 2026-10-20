import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(get_package_share_directory('g1_sensor_restamp'), 'config', 'restamp.yaml')
    return LaunchDescription([
        Node(package='g1_sensor_restamp', executable='restamp_node', name='g1_sensor_restamp',
             parameters=[config], output='screen'),
    ])
