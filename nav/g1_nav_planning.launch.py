#!/usr/bin/env python3
"""Foxy port of the tested X2 NavigateToPose stack for G1."""

from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
import os


MAP_YAML = os.environ.get("G1_NAV_MAP_YAML", "/data/maps/teaching_lab_nav/teaching_lab_nav.yaml")
PARAMS = "/opt/g1_slam/config/g1_nav2_planning.yaml"
NAV_DIR = "/opt/g1_slam/nav"


def generate_launch_description():
    pose_bridge = ExecuteProcess(
        cmd=[
            "python3",
            NAV_DIR + "/g1_nav_pose_bridge.py",
            "--ros-args",
            "-r", "tf:=/tf",
            "-r", "tf_static:=/tf_static",
        ],
        output="screen",
    )
    map_server = Node(
        package="nav2_map_server",
        executable="map_server",
        namespace="g1_nav",
        name="map_server",
        output="screen",
        parameters=[PARAMS, {"yaml_filename": MAP_YAML, "use_sim_time": False}],
    )
    planner_server = Node(
        package="nav2_planner",
        executable="planner_server",
        namespace="g1_nav",
        name="planner_server",
        output="screen",
        parameters=[PARAMS],
        # Foxy Navfn leaves the per-pose frame IDs empty. Keep its raw display
        # topic separate; the controller still receives the action result.
        remappings=[("plan", "/g1_nav/plan_raw")],
    )
    # Any possible velocity output is quarantined on a preview topic.
    controller_server = Node(
        package="nav2_controller",
        executable="controller_server",
        namespace="g1_nav",
        name="controller_server",
        output="screen",
        parameters=[PARAMS],
        remappings=[("cmd_vel", "/g1_nav/cmd_vel_preview")],
    )
    recoveries_server = Node(
        package="nav2_recoveries",
        executable="recoveries_server",
        namespace="g1_nav",
        name="recoveries_server",
        output="screen",
        parameters=[PARAMS],
    )
    bt_navigator = Node(
        package="nav2_bt_navigator",
        executable="bt_navigator",
        namespace="g1_nav",
        name="bt_navigator",
        output="screen",
        parameters=[PARAMS],
    )
    lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        namespace="g1_nav",
        name="lifecycle_manager_planning",
        output="screen",
        parameters=[{
            "use_sim_time": False,
            "autostart": True,
            "node_names": [
                "map_server",
                "planner_server",
                "controller_server",
                "recoveries_server",
                "bt_navigator",
            ],
        }],
    )
    goal_to_nav = ExecuteProcess(
        cmd=["python3", NAV_DIR + "/g1_goal_to_nav.py"], output="screen"
    )
    path_frame_relay = ExecuteProcess(
        cmd=["python3", NAV_DIR + "/g1_path_frame_relay.py"], output="screen"
    )
    lidar_costmap_relay = ExecuteProcess(
        cmd=["python3", NAV_DIR + "/g1_lidar_costmap_relay.py"], output="screen"
    )
    return LaunchDescription([
        pose_bridge,
        path_frame_relay,
        lidar_costmap_relay,
        map_server,
        TimerAction(period=1.0, actions=[planner_server]),
        TimerAction(period=2.0, actions=[controller_server]),
        TimerAction(period=3.0, actions=[recoveries_server]),
        TimerAction(period=4.0, actions=[bt_navigator]),
        TimerAction(period=5.0, actions=[lifecycle_manager]),
        TimerAction(period=7.0, actions=[goal_to_nav]),
    ])
