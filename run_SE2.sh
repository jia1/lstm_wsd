#!/bin/bash

docker build -t kageback/wsd . && \
docker run -t --rm kageback/wsd /bin/bash -c "python -u train.py 2;
					      python -u eval.py 2;
				              python -u score.py ./tmp/result ./data/senseval2/Senseval2.key ./data/senseval2/sensemap"


