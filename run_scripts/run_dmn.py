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

tasks = [6, 1, 2, 3]

for task in tasks:
  for i in xrange(5):
    # Generate random parameters in range
    reg = 0
    if i > 0:
      reg = np.random.uniform(1e-6,1e-4,1)[0]

    # These parameters will be passed to dynamic_memory_network.py.
    parameters = " -reg " + str(reg) + " -task " + str(task)
    command = "/usr/bin/expect -f run_dmn.exp %s '%s' &" \
      % (get_server_number(counter), parameters)
    print 'Executing command:', command
    os.system(command)

    counter += 1
    time.sleep(5)
