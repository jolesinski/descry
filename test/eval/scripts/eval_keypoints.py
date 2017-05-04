#!/usr/bin/python

import yaml, subprocess, sys

iss_yaml = """
model:
    name: object_10
    keypoints:
        type: iss-cupcl
        salient-radius: 0.01
        non-max-radius: 0.02
        lambda-ratio-21: 0.975
        lambda-ratio-32: 0.975
        lambda-threshold-3: 5e-07
        min-neighbours: 30

scene:
    name: T_01_willow_dataset
    keypoints:
        type: iss-cupcl
        salient-radius: 0.01
        non-max-radius: 0.02
        lambda-ratio-21: 0.975
        lambda-ratio-32: 0.975
        lambda-threshold-3: 5e-07
        min-neighbours: 30

metrics:
    radius: 0.01
"""

iss_conf = yaml.load(iss_yaml)

for incr in range(3, 4):
    iss_conf['model']['keypoints']['salient-radius'] += 0.005
    print "Model keys salient radius " + str(iss_conf['model']['keypoints']['salient-radius'])
    sys.stdout.flush()
    with open('/tmp/eval_keys.yaml', 'w') as outfile:
        yaml.dump(iss_conf, outfile, default_flow_style=False)

    subprocess.check_call(['test/eval/descry_key_eval', '/tmp/eval_keys.yaml'])
