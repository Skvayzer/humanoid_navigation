#!/usr/bin/env python3
"""Bake a measured floor plane into a binary PCD as a ROS Z-up map.

The input plane is z = a*x + b*y + c.  The physical free-space normal is
chosen opposite raw +Z because this G1 recording has the floor above the
sensor origin.  Raw +X is projected onto the floor to fix the otherwise
ambiguous map yaw.
"""

import argparse
import math
from pathlib import Path

import numpy as np


EXPECTED_FIELDS = [
    "x", "y", "z", "intensity", "normal_x", "normal_y", "normal_z", "curvature"
]


def rotation_to_quaternion(rotation):
    """Return ROS quaternion x,y,z,w for a proper 3x3 rotation matrix."""
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / s
            qx = 0.25 * s
            qy = (rotation[0, 1] + rotation[1, 0]) / s
            qz = (rotation[0, 2] + rotation[2, 0]) / s
        elif index == 1:
            s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / s
            qx = (rotation[0, 1] + rotation[1, 0]) / s
            qy = 0.25 * s
            qz = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / s
            qx = (rotation[0, 2] + rotation[2, 0]) / s
            qy = (rotation[1, 2] + rotation[2, 1]) / s
            qz = 0.25 * s
    quaternion = np.array([qx, qy, qz, qw], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return quaternion


def read_binary_pcd(path):
    header = []
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("PCD header ended before DATA")
            header.append(line)
            if line.startswith(b"DATA "):
                if line.strip() != b"DATA binary":
                    raise ValueError("only DATA binary PCD files are supported")
                break
        payload = stream.read()

    values = {}
    for line in header:
        parts = line.decode("ascii").strip().split()
        if parts:
            values[parts[0]] = parts[1:]
    if values.get("FIELDS") != EXPECTED_FIELDS:
        raise ValueError("unexpected PCD fields: {}".format(values.get("FIELDS")))
    if values.get("SIZE") != ["4"] * 8 or values.get("TYPE") != ["F"] * 8:
        raise ValueError("expected eight float32 fields")
    points = int(values["POINTS"][0])
    cloud = np.frombuffer(payload, dtype="<f4").copy()
    if cloud.size != points * 8:
        raise ValueError("payload size does not match POINTS")
    return header, cloud.reshape(points, 8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--a", type=float, required=True)
    parser.add_argument("--b", type=float, required=True)
    parser.add_argument("--c", type=float, required=True)
    args = parser.parse_args()

    header, cloud = read_binary_pcd(args.input)
    scale = math.sqrt(1.0 + args.a * args.a + args.b * args.b)
    up_raw = np.array([args.a, args.b, -1.0], dtype=np.float64) / scale
    raw_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    aligned_x_raw = raw_x - up_raw * np.dot(raw_x, up_raw)
    aligned_x_raw /= np.linalg.norm(aligned_x_raw)
    aligned_y_raw = np.cross(up_raw, aligned_x_raw)
    rotation = np.vstack((aligned_x_raw, aligned_y_raw, up_raw))
    translation = np.array([0.0, 0.0, args.c / scale], dtype=np.float64)

    finite_xyz = np.isfinite(cloud[:, :3]).all(axis=1)
    cloud[finite_xyz, :3] = (
        cloud[finite_xyz, :3].astype(np.float64) @ rotation.T + translation
    ).astype("<f4")
    finite_normals = np.isfinite(cloud[:, 4:7]).all(axis=1)
    nonzero_normals = np.linalg.norm(cloud[:, 4:7], axis=1) > 0.0
    rotate_normals = finite_normals & nonzero_normals
    cloud[rotate_normals, 4:7] = (
        cloud[rotate_normals, 4:7].astype(np.float64) @ rotation.T
    ).astype("<f4")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        stream.writelines(header)
        stream.write(cloud.tobytes(order="C"))

    quaternion = rotation_to_quaternion(rotation)
    transformed_floor_z = (
        up_raw @ np.array([0.0, 0.0, args.c], dtype=np.float64) + translation[2]
    )
    print("rotation_rows={}".format(np.array2string(rotation, precision=9)))
    print("translation={}".format(np.array2string(translation, precision=9)))
    print("quaternion_xyzw={}".format(np.array2string(quaternion, precision=9)))
    print("mapping_origin_height_m={:.9f}".format(translation[2]))
    print("floor_check_z_m={:.9f}".format(transformed_floor_z))
    print("points={}".format(cloud.shape[0]))


if __name__ == "__main__":
    main()
