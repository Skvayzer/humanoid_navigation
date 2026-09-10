#!/usr/bin/env python3
"""Download/hash-check a pinned CAT model; optional OFFLINE graph smoke test.

No ROS, robot SDK, actuator code, automatic dependency installation or live input.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import urllib.request

MANIFEST = Path(__file__).with_name('checkpoint_manifest.json')


def verify(directory, manifest=None):
    manifest = manifest or json.loads(MANIFEST.read_text())
    for name, item in manifest['files'].items():
        content = (Path(directory)/name).read_bytes()
        if len(content) != item['size'] or hashlib.sha256(content).hexdigest() != item['sha256']:
            raise ValueError('checkpoint hash/size mismatch: '+name)
    cfg = json.loads((Path(directory)/'config.json').read_text())
    for key, value in manifest['expected'].items():
        if cfg['env_config'][key] != value:
            raise ValueError('checkpoint contract mismatch: '+key)
    return dict(name=manifest['name'], artifact_verified=True, deployment_approved=False,
                inference_enabled=False, expected=manifest['expected'])


def fetch(directory):
    manifest = json.loads(MANIFEST.read_text())
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, item in manifest['files'].items():
        target = directory/name
        if target.exists():
            content = target.read_bytes()
            if hashlib.sha256(content).hexdigest() != item['sha256']:
                raise ValueError('refusing to overwrite different existing file: '+str(target))
            continue
        url = 'https://huggingface.co/{}/resolve/{}/{}'.format(
            manifest['repository'], manifest['revision'], item['path'])
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read(item['size']+1)
        if len(content) != item['size'] or hashlib.sha256(content).hexdigest() != item['sha256']:
            raise ValueError('download verification failed: '+name)
        with tempfile.NamedTemporaryFile(dir=str(directory), delete=False) as tmp:
            temporary = Path(tmp.name)
            tmp.write(content)
        try:
            # Atomic create, never overwrite a concurrently created artifact.
            os.link(str(temporary), str(target))
        finally:
            temporary.unlink()
    return verify(directory)


def smoke(directory):
    result = verify(directory)
    import numpy as np
    import onnx
    import onnxruntime as ort
    model = onnx.load(str(Path(directory)/'policy.onnx'))
    onnx.checker.check_model(model)
    if any(x.data_location == onnx.TensorProto.EXTERNAL for x in model.graph.initializer):
        raise ValueError('external ONNX data not allowed')
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(model.SerializeToString(), options, providers=['CPUExecutionProvider'])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    def compatible(shape, width):
        return (len(shape) == 2 and shape[1] == width and
                (shape[0] == 1 or shape[0] is None or isinstance(shape[0], str)))
    if len(inputs) != 1 or inputs[0].type != 'tensor(float)' or not compatible(inputs[0].shape, 162):
        raise ValueError('unexpected ONNX input contract: '+str([(i.name, i.shape) for i in inputs]))
    if len(outputs) != 1 or not compatible(outputs[0].shape, 12):
        raise ValueError('unexpected ONNX output contract')
    timings = []
    for _ in range(25):
        start = time.perf_counter()
        output = session.run(None, {inputs[0].name: np.zeros((1, 162), np.float32)})[0]
        timings.append((time.perf_counter()-start)*1000)
        if output.shape != (1, 12) or not np.isfinite(output).all():
            raise ValueError('nonfinite synthetic inference result')
    result.update(offline_smoke_only=True, input_name=inputs[0].name, input_shape=inputs[0].shape,
                  output_name=outputs[0].name, output_shape=outputs[0].shape,
                  median_inference_ms=float(np.median(timings[5:])))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['fetch', 'verify', 'smoke'])
    parser.add_argument('--directory', default=str(Path(__file__).parent/'models/generalist_v1'))
    args = parser.parse_args()
    print(json.dumps(dict(fetch=fetch, verify=verify, smoke=smoke)[args.action](args.directory), indent=2))
