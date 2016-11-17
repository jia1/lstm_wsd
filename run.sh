#!/bin/bash

docker build -t kageback/wsd . && \
docker run -t --rm kageback/wsd python -u train.py && \
				 python -u eval.py && \
				 python -u score.py ./tmp/result ./data/senseval2/Senseval2.key ./data/senseval2/sensemap

