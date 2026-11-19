#!/usr/bin/env python3
"""Foxy -> Humble MarkerArray relay for the viewer (runs in g1-rr-foxglove next to domain_bridge and foxglove_bridge).

visualization_msgs/Marker grew fields in Humble (texture_resource, texture, uv_coordinates, mesh_file), so the robot's
Foxy bytes cannot be passed through as a Humble MarkerArray: Foxglove (given Humble's schema by foxglove_bridge) reads
past their end ("Offset is outside the bounds of the DataView"). This node takes the robot's MarkerArray topics as raw
Foxy CDR in domain 0, decodes them with the Foxy layout (demo/bagread.py, the same reader the take renderer uses) and
republishes real Humble messages in the viewer's domain 7 (latched, as the executor publishes them).

  marker_relay.py /g1_wb/virtual_scene [...]
"""
import sys

import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

from bagread import Cdr

ROBOT_DOMAIN, VIEWER_DOMAIN = 0, 7
LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST)


def foxy_markers(raw):
    """MarkerArray in Foxy's CDR layout -> Humble MarkerArray."""
    r = Cdr(raw); out = MarkerArray()
    for _ in range(r.u32()):
        m = Marker()
        sec, nsec = r.take('iI', 4); m.header.frame_id = r.string()
        m.header.stamp.sec, m.header.stamp.nanosec = sec, nsec
        m.ns = r.string(); m.id = r.i32(); m.type = r.i32(); m.action = r.i32()
        p = r.f64(3); q = r.f64(4)
        m.pose.position.x, m.pose.position.y, m.pose.position.z = p
        m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = q
        m.scale.x, m.scale.y, m.scale.z = r.f64(3)
        m.color.r, m.color.g, m.color.b, m.color.a = r.f32(4)
        m.lifetime.sec, m.lifetime.nanosec = r.take('iI', 4)
        m.frame_locked = bool(r.u8())
        m.points = [Point(x=x, y=y, z=z) for x, y, z in (r.f64(3) for _ in range(r.u32()))]
        m.colors = [ColorRGBA(r=c[0], g=c[1], b=c[2], a=c[3]) for c in (r.f32(4) for _ in range(r.u32()))]
        m.text = r.string(); m.mesh_resource = r.string(); m.mesh_use_embedded_materials = bool(r.u8())
        out.markers.append(m)
    return out


def main():
    topics = sys.argv[1:] or ['/g1_wb/virtual_scene']
    robot, viewer = Context(), Context()
    rclpy.init(context=robot, domain_id=ROBOT_DOMAIN); rclpy.init(context=viewer, domain_id=VIEWER_DOMAIN)
    sub_node = rclpy.create_node('marker_relay_in', context=robot)
    pub_node = rclpy.create_node('marker_relay_out', context=viewer)
    log = sub_node.get_logger()
    for t in topics:
        pub = pub_node.create_publisher(MarkerArray, t, LATCHED)

        def relay(raw, pub=pub, t=t):
            try:
                pub.publish(foxy_markers(raw))
            except Exception as exc:  # a malformed message must not stop the relay
                log.warn('%s: %r' % (t, exc))
        sub_node.create_subscription(MarkerArray, t, relay, LATCHED, raw=True)
    log.info('relaying %s (Foxy MarkerArray, domain %d) -> domain %d' % (', '.join(topics), ROBOT_DOMAIN, VIEWER_DOMAIN))
    ex = SingleThreadedExecutor(context=robot); ex.add_node(sub_node)
    try:
        ex.spin()
    finally:
        sub_node.destroy_node(); pub_node.destroy_node(); rclpy.shutdown(context=robot); rclpy.shutdown(context=viewer)


if __name__ == '__main__':
    main()
