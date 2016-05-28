#/usr/bin/python
#
# File: run_dmn.py
# ---------------------
# Parallelize runs of a program across multiple Stanford corn machines.
# This script generates parameter combinations and hands off to
# run_dmn.exp.
#
# This script should be run from within screen on a corn server.
# > ssh SUNetID@corn.stanford.edu
# > cd path/to/repo/scripts
# > screen
# > python run_dmn.py
# > # You can press "ctrl-a d" to detach from screen and "ctrl-a r" to re-attach.

import os
import time
import numpy as np
import random

def get_server_number(counter):
  # There are 30 corn servers, named corn01 through corn31.
  return '%02d' % (counter % 30 + 1)

# Keeps track of which corn server to use.
counter = 0

# Generate random parameters in range
lrs = np.random.uniform(1e-4,1e-1,3)
regs = np.random.uniform(1e-6,1e-1,3)
#dropout_rates = np.random.uniform(0,0.3,1)

for lr in lrs:
  for reg in regs:
    # These parameters will be passed to dynamic_memory_network.py.
    parameters = "-lr " + str(lr) + " -reg " + str(reg)
    command = "/usr/bin/expect -f run_dmn.exp %s '%s' &" \
      % (get_server_number(counter), parameters)
    print 'Executing command:', command
    os.system(command)
  counter += 1
  time.sleep(5)
