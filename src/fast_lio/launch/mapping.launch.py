import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def mapping_node(context):
    # outdoor:=true loads config/outdoor.yaml after the main parameter file (larger voxels).
    parameters = [LaunchConfiguration('config').perform(context)]
    if LaunchConfiguration('outdoor').perform(context).lower() in ('true', '1'):
        parameters.append(os.path.join(get_package_share_directory('fast_lio'), 'config', 'outdoor.yaml'))
    parameters.append({'use_sim_time': LaunchConfiguration('use_sim_time')})
    return [Node(
        package='fast_lio',
        executable='fastlio_mapping',
        parameters=parameters,
        output='screen',
    )]


def generate_launch_description():
    default_config = os.path.join(get_package_share_directory('fast_lio'), 'config', 'g1_mid360.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=default_config,
                              description='Full path of the parameter file'),
        DeclareLaunchArgument('outdoor', default_value='false',
                              description='Load config/outdoor.yaml after the parameter file (larger voxels)'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        OpaqueFunction(function=mapping_node),
    ])
