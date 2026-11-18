#!/usr/bin/env python3
"""wholebody_layout_v3.json from wholebody_layout_v2.1.json (hand-tuned in Foxglove; never edited here).

Changes: the "G1 controls" panel gets the scenes with their geometry (a drop-down with top-view previews) and a short note
each; the scene list is ordered simple -> hard; the ESDF slice script is woken by /tf, not odometry. Needs g1_panels/g1-wholebody-panels.foxe >= 1.1.0.

  python3 make_layout_v3.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCENE_DIRS = [HERE.parent / 'scenes', HERE.parents[1] / 'cat' / 'wholebody' / 'scenes']
ORDER = ['flat_walk_4m', 'hurdle_low_3m', 'beam_high_3m', 'beam_low_3m', 'gap_044_exact', 'sidegap_plain_047',
         'sidegap_corridor_048', 'course_mixed_25m']
CONTROLS_ID = 'g1-wholebody-panels.G1 controls!controls'


def note(d):
    boxes = d['boxes']
    if not boxes:
        return 'open floor, goal %.1f m' % d['goal'][0]
    cats = [b.get('category', '') for b in boxes]
    if len(boxes) > 6:
        return '%.0f m course: %d obstacles' % (d['goal'][0], len(boxes))
    b = boxes[0]; top, bottom = b['center'][2] + b['half_size'][2], b['center'][2] - b['half_size'][2]
    if cats[0] == 'hurdle':
        return 'step over a %.2f m hurdle at %.1f m' % (top, b['center'][0])
    if cats[0] == 'beam':
        return 'duck under a beam, bottom %.2f m, at %.1f m' % (bottom, b['center'][0])
    ys = sorted(x['center'][1] - x['half_size'][1] for x in boxes if x['center'][1] > 0)
    yl = sorted(x['center'][1] + x['half_size'][1] for x in boxes if x['center'][1] < 0)
    gap = (ys[0] - yl[-1]) if ys and yl else None
    return 'gap %.2f m between walls at %.1f m' % (gap, b['center'][0]) if gap else '%d boxes' % len(boxes)


def scene(name):
    path = next(d / (name + '.json') for d in SCENE_DIRS if (d / (name + '.json')).exists())
    d = json.loads(path.read_text())
    r = lambda v: [round(float(x), 3) for x in v]
    return dict(name=name, note=note(d), goal=r(d['goal'][:2]),
                boxes=[dict(c=r(b['center']), h=r(b['half_size']), yaw=round(float(b.get('yaw', 0.)), 4)) for b in d['boxes']])


v3 = json.loads((HERE / 'wholebody_layout_v2.1.json').read_text())
names = [n for n in ORDER if any((d / (n + '.json')).exists() for d in SCENE_DIRS)]
v3['configById'][CONTROLS_ID]['scenes'] = [scene(n) for n in names]
# the ESDF slice script is woken by /tf (55 Hz, the 3D panel subscribes to it anyway) instead of /g1_slam/odometry
# (50 Hz JSON through rosbridge only for that)
for node in v3.get('userNodes', {}).values():
    node['sourceCode'] = node['sourceCode'].replace('"/g1_slam/odometry"', '"/tf"').replace('Input<"/g1_slam/odometry">', 'Input<"/tf">') \
        .replace('Odometry (50 Hz) only wakes the script', '/tf (~55 Hz) only wakes the script')
(HERE / 'wholebody_layout_v3.json').write_text(json.dumps(v3, indent=1, ensure_ascii=False) + '\n')
for s in v3['configById'][CONTROLS_ID]['scenes']:
    print('  %-22s %s' % (s['name'], s['note']))
