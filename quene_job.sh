#!/bin/bash

# Create log dir, if it doesn't aready exist.
[ -d log ] || mkdir log 

# Add job to queue
qsub -cwd \
  -e ./log/run.sh.error \
  -o ./log/run.sh.log \
  ./run.sh

