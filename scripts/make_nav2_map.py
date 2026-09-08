#!/usr/bin/env python3
"""Create a conservative Nav2 occupancy grid from the aligned G1 PCD map."""

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def load_xyz(path):
    header = {}
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("invalid PCD header")
            parts = line.decode("ascii").strip().split()
            if parts:
                header[parts[0]] = parts[1:]
            if parts[:1] == ["DATA"]:
                if parts[1:] != ["binary"]:
                    raise ValueError("only binary PCD is supported")
                payload = stream.read()
                break
    fields = header["FIELDS"]
    if header["SIZE"] != ["4"] * len(fields) or header["TYPE"] != ["F"] * len(fields):
        raise ValueError("expected float32 PCD fields")
    points = int(header["POINTS"][0])
    cloud = np.frombuffer(payload, dtype="<f4").reshape(points, len(fields))
    xyz = cloud[:, [fields.index("x"), fields.index("y"), fields.index("z")]]
    return xyz[np.isfinite(xyz).all(axis=1)].astype(np.float64)


def disk(radius):
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return x * x + y * y <= radius * radius


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--padding", type=float, default=0.50)
    parser.add_argument("--floor-min", type=float, default=-0.05)
    parser.add_argument("--floor-max", type=float, default=0.08)
    parser.add_argument("--obstacle-min", type=float, default=0.12)
    parser.add_argument("--obstacle-max", type=float, default=1.70)
    parser.add_argument("--free-support-radius", type=float, default=0.15)
    parser.add_argument("--obstacle-seal-radius", type=float, default=0.10)
    args = parser.parse_args()

    xyz = load_xyz(args.input)
    xmin = math.floor((xyz[:, 0].min() - args.padding) / args.resolution) * args.resolution
    ymin = math.floor((xyz[:, 1].min() - args.padding) / args.resolution) * args.resolution
    xmax = math.ceil((xyz[:, 0].max() + args.padding) / args.resolution) * args.resolution
    ymax = math.ceil((xyz[:, 1].max() + args.padding) / args.resolution) * args.resolution
    width = int(round((xmax - xmin) / args.resolution)) + 1
    height = int(round((ymax - ymin) / args.resolution)) + 1

    def cells(points):
        cols = np.floor((points[:, 0] - xmin) / args.resolution).astype(np.int64)
        rows_from_bottom = np.floor((points[:, 1] - ymin) / args.resolution).astype(np.int64)
        rows = height - 1 - rows_from_bottom
        valid = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
        return rows[valid], cols[valid]

    floor_points = xyz[(xyz[:, 2] >= args.floor_min) & (xyz[:, 2] <= args.floor_max)]
    obstacle_points = xyz[
        (xyz[:, 2] >= args.obstacle_min) & (xyz[:, 2] <= args.obstacle_max)
    ]
    floor_hits = np.zeros((height, width), dtype=bool)
    obstacle_hits = np.zeros((height, width), dtype=bool)
    floor_rows, floor_cols = cells(floor_points)
    obstacle_rows, obstacle_cols = cells(obstacle_points)
    floor_hits[floor_rows, floor_cols] = True
    obstacle_hits[obstacle_rows, obstacle_cols] = True

    free_radius_cells = max(1, int(round(args.free_support_radius / args.resolution)))
    obstacle_radius_cells = max(1, int(round(args.obstacle_seal_radius / args.resolution)))
    free = ndimage.binary_dilation(floor_hits, structure=disk(free_radius_cells))
    free = ndimage.binary_closing(free, structure=disk(free_radius_cells))
    occupied = ndimage.binary_dilation(
        obstacle_hits, structure=disk(obstacle_radius_cells)
    )
    free &= ~occupied

    # ROS trinary map: black occupied, white free, gray unknown.
    image = np.full((height, width), 205, dtype=np.uint8)
    image[free] = 254
    image[occupied] = 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.output_dir / "teaching_lab_nav.pgm"
    yaml_path = args.output_dir / "teaching_lab_nav.yaml"
    Image.fromarray(image, mode="L").save(str(image_path))
    yaml_path.write_text(
        "image: teaching_lab_nav.pgm\n"
        "resolution: {:.6f}\n".format(args.resolution)
        + "origin: [{:.6f}, {:.6f}, 0.0]\n".format(xmin, ymin)
        + "negate: 0\n"
        + "occupied_thresh: 0.65\n"
        + "free_thresh: 0.196\n"
        + "mode: trinary\n",
        encoding="utf-8",
    )
    preview = np.zeros((height, width, 3), dtype=np.uint8)
    preview[image == 205] = [110, 110, 110]
    preview[image == 254] = [255, 255, 255]
    preview[image == 0] = [0, 0, 0]
    Image.fromarray(preview, mode="RGB").save(str(args.output_dir / "teaching_lab_nav_preview.png"))

    total = width * height
    print("grid={}x{} resolution={:.3f} origin=({:.3f},{:.3f})".format(
        width, height, args.resolution, xmin, ymin
    ))
    print("source floor_points={} obstacle_points={}".format(
        len(floor_points), len(obstacle_points)
    ))
    print("cells free={} ({:.1f}%) occupied={} ({:.1f}%) unknown={} ({:.1f}%)".format(
        int(free.sum()), 100.0 * free.sum() / total,
        int(occupied.sum()), 100.0 * occupied.sum() / total,
        int((image == 205).sum()), 100.0 * (image == 205).sum() / total,
    ))


if __name__ == "__main__":
    main()
