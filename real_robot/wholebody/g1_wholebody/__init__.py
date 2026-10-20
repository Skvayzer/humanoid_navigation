"""Real-robot runtime for the CAT whole-body policy on our Unitree G1 (feature/g1-real-robot).

Middleware-free core (lowlevel, frames, fields, safety, runner, fake_robot) + thin ROS 2 adapters (ros_*).
The core is what the robot runs; the same core runs closed-loop against MuJoCo on the laptop.
"""
