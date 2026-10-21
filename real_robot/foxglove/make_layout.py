#!/usr/bin/env python3
"""Writes wholebody_layout.json (Foxglove, Rosbridge ws://localhost:9090): one view for shadow and active runs.

  python3 make_layout.py [extra scene names ...]     (a button per scene: the shipped ones + the names given)

3D (display frame `map`, the floor plane: a click with Publish > Pose sends /g1_wb/goal_pose on the floor):
the room from SLAM in grey, the virtual boxes, the ESDF slice the policy sees, its body sites, the route, the
geofence, the pelvis. Buttons publish /g1_wb/operator, /g1_wb/obstacle_mode and /g1_wb/scene; indicators show the
executor mode, whether the rope may be slackened, and which obstacles the fields are made of.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCENES = sorted(p.stem for p in (HERE.parents[1] / 'cat' / 'wholebody' / 'scenes').glob('*.json')) + sys.argv[1:]

SCRIPT = r'''// Splits the executor's JSON status (/g1_wb/status, 5 Hz) into fields for the Indicator and Plot panels.
import { Input } from "./types";

export const inputs = ["/g1_wb/status"];
export const output = "/studio_script/wb_status";

type Output = {
  mode: string;
  rope: string;
  obstacles: string;
  scene: string;
  goal: string;
  reason: string;
  policy_wait: string;
  step_ms: number;
  latency_ms: number;
  overruns: number;
  fields_age_s: number;
  field_solve_s: number;
  field_error: string;
  goal_dist_m: number;
  waypoint: number;
  latch: number;
  reached: boolean;
  hold: boolean;
  active_allowed: boolean;
};

const num = (v: unknown): number => (typeof v === "number" ? v : NaN);

export default function script(event: Input<"/g1_wb/status">): Output | undefined {
  let s: Record<string, unknown>;
  try {
    s = JSON.parse(event.message.data);
  } catch {
    return undefined;
  }
  const mode = String(s.mode ?? "?");
  const wait = s.policy_wait == null ? "" : String(s.policy_wait);
  const xy = s.pelvis_xy as number[] | null;
  const fg = s.field_goal as number[] | null;
  const g = s.goal_xy as number[] | null;
  // the gantry procedure: slacken the rope only once the policy itself stands still (training stop latch at 0)
  let rope = "motors off";
  if (mode === "STAND_UP" || mode === "STAND") rope = "KEEP ROPE: stiff stand";
  else if (mode === "POLICY" && wait !== "") rope = "KEEP ROPE: policy waiting";
  else if (mode === "POLICY" && s.hold) rope = s.latch === 0 ? "standing: rope may slacken" : "KEEP ROPE: settling";
  else if (mode === "POLICY") rope = "walking";
  else if (mode === "DAMP") rope = "DAMP";
  return {
    mode, rope, obstacles: String(s.obstacles ?? "?"), scene: String(s.scene ?? "?"),
    goal: g ? `(${g[0].toFixed(2)}, ${g[1].toFixed(2)})` : "scene goal",
    reason: String(s.reason ?? ""), policy_wait: wait,
    step_ms: num(s.step_ms), latency_ms: num(s.latency_ms), overruns: num(s.overruns),
    fields_age_s: num(s.fields_age_s), field_solve_s: num(s.field_solve_s),
    field_error: s.field_error == null ? "" : String(s.field_error),
    goal_dist_m: xy && fg ? Math.hypot(xy[0] - fg[0], xy[1] - fg[1]) : NaN,
    waypoint: num(s.waypoint), latch: num(s.latch), reached: Boolean(s.reached), hold: Boolean(s.hold),
    active_allowed: Boolean(s.active_allowed),
  };
}
'''

GREY, ORANGE, GREEN, BLUE, RED, AMBER = '#707070', '#e0701a', '#1f9d3a', '#2f6fdf', '#e02020', '#d08000'


def rule(value, color, label=None):
    return dict(operator='=', rawValue=value, color=color, label=label or value)


def path(field, label):
    return dict(value='/studio_script/wb_status.' + field, enabled=True, timestampMethod='receiveTime', label=label)


def plot(paths, ymax):
    return dict(paths=paths, minYValue=0, maxYValue=ymax, showXAxisLabels=True, showYAxisLabels=True, showLegend=True,
                legendDisplay='floating', showPlotValuesInLegend=True, isSynced=True, xAxisVal='timestamp', followingViewWidth=60)


def button(topic, data, text, color, tip):
    return dict(topicName=topic, datatype='std_msgs/msg/String', buttonText=text, buttonTooltip=tip, buttonColor=color,
                advancedView=False, value=json.dumps({'data': data}))


def row(ids, direction='row'):
    """Equal panes in a mosaic: nested halves."""
    if len(ids) == 1:
        return ids[0]
    k = len(ids) // 2
    return dict(direction=direction, first=row(ids[:k], direction), second=row(ids[k:], direction), splitPercentage=100 * k / len(ids))


config = {
    '3D!wb': {
        'cameraState': dict(distance=8, perspective=True, phi=55, thetaOffset=-60, targetOffset=[1.5, 0, 0], target=[0, 0, 0],
                            targetOrientation=[0, 0, 0, 1], fovy=45, near=0.01, far=500),
        'followMode': 'follow-pose', 'followTf': 'map',
        'scene': {'enableStats': False},
        'transforms': {'frame:camera_init': {'visible': False}, 'frame:body': {'visible': False}},
        'publish': {'type': 'pose', 'poseTopic': '/g1_wb/goal_pose', 'pointTopic': '/clicked_point',
                    'poseEstimateTopic': '/initialpose', 'poseEstimateXDeviation': 0.5, 'poseEstimateYDeviation': 0.5,
                    'poseEstimateThetaDeviation': 0.26},
        'topics': {
            # the real room (SLAM, 10 cm voxels) in grey; the live obstacle cloud of 'real' mode is heavier: off by default
            '/g1_slam/map': {'visible': True, 'pointSize': 2.5, 'pointShape': 'square', 'decayTime': 0, 'colorMode': 'flat',
                             'flatColor': '#9a9a9acc'},
            '/g1_slam/obstacles': {'visible': False, 'pointSize': 3, 'pointShape': 'square', 'decayTime': 0, 'colorMode': 'flat',
                                   'flatColor': '#505050ff'},
            # what the policy sees: the virtual boxes (esdf_map_virtual), the ESDF slice at 0.75 m, its body sites
            '/g1_wb/virtual_scene': {'visible': True},
            '/g1_wb/sdf_slice': {'visible': True, 'pointSize': 4, 'pointShape': 'square', 'decayTime': 0, 'colorField': 'intensity',
                                 'colorMode': 'colormap', 'colorMap': 'turbo', 'minValue': -0.1, 'maxValue': 1.2, 'explicitAlpha': 0.6},
            '/g1_wb/sites': {'visible': True, 'pointSize': 14, 'pointShape': 'circle', 'decayTime': 0, 'colorField': 'sdf',
                             'colorMode': 'gradient', 'gradient': ['#ff2020ff', '#20e050ff'], 'minValue': 0.0, 'maxValue': 0.3},
            '/g1_wb/pelvis': {'visible': True, 'type': 'arrow', 'arrowScale': [0.45, 0.08, 0.08], 'color': '#2f9bffff'},
            '/g1_wb/goal': {'visible': True, 'type': 'arrow', 'arrowScale': [0.3, 0.12, 0.12], 'color': '#20e050ff'},
            '/g1_wb/goal_pose': {'visible': True, 'type': 'arrow', 'arrowScale': [0.4, 0.1, 0.1], 'color': '#e0d020ff'},
            '/g1_slam/cloud_registered': {'visible': False},
            '/g1_cat/occupancy_cat': {'visible': False, 'pointSize': 4, 'decayTime': 2.5, 'colorMode': 'flat', 'flatColor': '#ff3df5ff'},
        },
        'layers': {'grid': {'layerId': 'foxglove.Grid', 'visible': True, 'frameId': 'map', 'size': 12, 'divisions': 12}},
    },
    'Indicator!mode': {
        'path': '/studio_script/wb_status.mode', 'style': 'background', 'fallbackColor': '#555555', 'fallbackLabel': 'no status',
        'rules': [rule('DAMP', RED), rule('POLICY', GREEN), rule('STAND', AMBER), rule('STAND_UP', AMBER), rule('SHADOW', BLUE),
                  rule('IDLE', GREY)]},
    'Indicator!rope': {
        'path': '/studio_script/wb_status.rope', 'style': 'background', 'fallbackColor': '#555555', 'fallbackLabel': '-',
        'rules': [rule('KEEP ROPE: stiff stand', AMBER), rule('KEEP ROPE: policy waiting', '#e05000'), rule('KEEP ROPE: settling', '#e05000'),
                  rule('standing: rope may slacken', GREEN), rule('walking', GREEN), rule('DAMP', RED), rule('motors off', GREY)]},
    'Indicator!obstacles': {
        'path': '/studio_script/wb_status.obstacles', 'style': 'background', 'fallbackColor': '#555555', 'fallbackLabel': '?',
        'rules': [rule('virtual', ORANGE, 'obstacles: VIRTUAL'), rule('real', '#606060', 'obstacles: REAL'),
                  rule('both', BLUE, 'obstacles: VIRTUAL + REAL')]},
    'Plot!timing': plot([path('step_ms', 'step ms (budget 18)'), path('latency_ms', 'LowState -> command ms')], 25),
    'Plot!fields': plot([path('fields_age_s', 'fields age s'), path('field_solve_s', 'solve s'), path('goal_dist_m', 'pelvis to field goal m')], 8),
    'RawMessages!status': {'topicPath': '/studio_script/wb_status', 'diffEnabled': False, 'expansion': 'all'},
    'RosOut!log': {'searchTerms': ['g1_wb'], 'minLogLevel': 2, 'topicToRender': '/rosout'},
}
ops = [('stand', AMBER, 'IDLE -> STAND (remote L1+A)'), ('balance', GREEN, 'STAND -> policy, goal = here (L1+X)'),
       ('go', GREEN, 'walk to the goal (R1+A)'), ('stop', AMBER, 'goal = here (R1+B)'), ('damp', RED, 'DAMP (L2+B is faster)'),
       ('reset', GREY, 'DAMP -> IDLE (L1+Y)')]
op_ids = []
for name, color, tip in ops:
    pid = 'Publish!op-%s' % name; op_ids.append(pid); config[pid] = button('/g1_wb/operator', name, name, color, tip)
obs_ids = []
for name, color in (('virtual', ORANGE), ('real', '#606060'), ('both', BLUE)):
    pid = 'Publish!obstacles-%s' % name; obs_ids.append(pid)
    config[pid] = button('/g1_wb/obstacle_mode', name, 'obstacles: ' + name, color, 'what the fields are made of (only while standing)')
scene_ids = []
for name in SCENES:
    pid = 'Publish!scene-%s' % name.replace('_', '-'); scene_ids.append(pid)
    config[pid] = button('/g1_wb/scene', name, name, ORANGE, 'lay this scene out in front of the robot (only while standing)')
config['Tab!info'] = {'activeTabIdx': 0, 'tabs': [{'title': 'Status', 'layout': 'RawMessages!status'},
                                                  {'title': 'Timing', 'layout': 'Plot!timing'},
                                                  {'title': 'Fields', 'layout': 'Plot!fields'},
                                                  {'title': 'Log', 'layout': 'RosOut!log'}]}

side = dict(direction='column', splitPercentage=10,
            first=row(['Indicator!mode', 'Indicator!rope', 'Indicator!obstacles']),
            second=dict(direction='column', splitPercentage=12, first=row(op_ids),
                        second=dict(direction='column', splitPercentage=10, first=row(obs_ids),
                                    second=dict(direction='column', splitPercentage=14, first=row(scene_ids), second='Tab!info'))))
layout = {'configById': config, 'globalVariables': {}, 'userNodes': {'wb-status': {'name': 'wb_status', 'sourceCode': SCRIPT}},
          'playbackConfig': {'speed': 1}, 'layout': dict(direction='row', splitPercentage=62, first='3D!wb', second=side)}
(HERE / 'wholebody_layout.json').write_text(json.dumps(layout, indent=1, ensure_ascii=False) + '\n')
print('wrote %s: %d scene buttons' % (HERE / 'wholebody_layout.json', len(scene_ids)))
