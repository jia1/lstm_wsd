#!/usr/bin/env python2.7
# encoding: utf-8

import sys, os, time, re
from subprocess import Popen, PIPE

# Read in first 5 arguments.
instance = sys.argv[1]
specifics = sys.argv[2]
cutoff = int(float(sys.argv[3]) + 1)
runlength = int(sys.argv[4])
seed = int(sys.argv[5])

seed = seed if seed != -1 else 1234

# Read in parameter setting and build a dictionary mapping param_name to param_value.
params = sys.argv[6:]
configMap = dict((name, value) for name, value in zip(params[::2], params[1::2]))

# Construct the call string to Spear.
binary = "/home/salomons/project/wsd/hyper.py"
cmd = "%s --seed %d --model-stdout --dimacs %s --tmout %d" %(binary, seed, instance, cutoff)
for name, value in configMap.items():
    cmd += " -%s" % name
    raw = " %s" % value
    cmd += raw.replace("\'", '')

# Execute the call and track its runtime.
# print(cmd)
start_time = time.time()
io = Popen(cmd.split(" "), stdout=PIPE, stderr=PIPE)
(stdout_, stderr_) = io.communicate()
runtime = time.time() - start_time

# Very simply parsing of Spear's output. Note that in practice we would check the found solution to guard against bugs.

try:
    best = re.findall('NEW BEST: 0.\d+', stdout_)[-1].split(' ')[-1]
    status = 'SUCCESS'
except:
    best = 0
    status = "CRASHED"
    print stderr_


# Output result for SMAC.
print("Result for SMAC: %s, 0, 0, %f, %s" % (status, - float(best), str(seed)))
