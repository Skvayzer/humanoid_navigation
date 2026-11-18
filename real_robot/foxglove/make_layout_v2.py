#!/usr/bin/env python3
"""wholebody_layout_v2.json from the reference wholebody_layout_v1.json (hand-tuned in Foxglove; never edited here).

Changes: the stock Variable Slider -> the "ESDF height" panel (live slider, wheel, arrow keys), and every Publish
button panel -> one "G1 controls" panel (operator, obstacles, scenes, operator speed): both from the g1_panels
extension (g1_panels/g1-wholebody-panels.foxe, install it first). Everything else is v1's.

  python3 make_layout_v2.py
"""
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = 'g1-wholebody-panels'     # g1_panels/package.json displayName: a local extension's panel type is '<displayName>.<panel name>'
ESDF_ID, CONTROLS_ID = EXT + '.ESDF height!esdfz', EXT + '.G1 controls!controls'

v1 = json.loads((HERE / 'wholebody_layout_v1.json').read_text())
v2 = copy.deepcopy(v1); cfg = v2['configById']
scenes = [json.loads(c['value'])['data'] for k, c in v1['configById'].items() if k.startswith('Publish!scene-')]
slider = next((c for k, c in v1['configById'].items() if k.startswith('GlobalVariableSliderPanel!')), {})
props = slider.get('sliderProps', {})
for k in [k for k in cfg if k.startswith('Publish!') or k.startswith('GlobalVariableSliderPanel!')]:
    del cfg[k]
cfg[ESDF_ID] = dict(variable=slider.get('globalVariableName', 'esdf_z'), min=props.get('min', 0.02), max=props.get('max', 1.5),
                    step=props.get('step', 0.04))
cfg[CONTROLS_ID] = dict(scenes=scenes, speedMin=0.1, speedMax=0.6, speedStep=0.05)
# the lidar the fields use: 0.10-1.40 m, no gantry-rope column (executor /g1_wb/obstacles_band), coloured by height
cfg['3D!wb'].setdefault('topics', {})['/g1_wb/obstacles_band'] = dict(
    visible=True, pointSize=4, pointShape='square', decayTime=0, colorField='intensity', colorMode='colormap',
    colorMap='turbo', minValue=0.1, maxValue=1.4, explicitAlpha=0.9)


def leaves(node):
    return [node] if isinstance(node, str) else leaves(node['first']) + leaves(node['second'])


def rewrite(node):
    if isinstance(node, str):
        return ESDF_ID if node.startswith('GlobalVariableSliderPanel!') else node
    if all(l.startswith('Publish!') for l in leaves(node)):            # a block of button panels -> one panel
        return CONTROLS_ID
    return dict(node, first=rewrite(node['first']), second=rewrite(node['second']))


v2['layout'] = rewrite(v2['layout'])
assert not [l for l in leaves(v2['layout']) if l.startswith('Publish!')], 'button panels left in the tree'
(HERE / 'wholebody_layout_v2.json').write_text(json.dumps(v2, indent=1, ensure_ascii=False) + '\n')
print('wrote wholebody_layout_v2.json: %d panels (v1: %d), scenes %s' % (len(cfg), len(v1['configById']), ', '.join(scenes)))
