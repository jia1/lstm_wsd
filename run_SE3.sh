#!/bin/bash

docker build -t kageback/wsd . && \
docker run -t --rm kageback/wsd /bin/bash -c "python -u train.py;
					      python -u eval.py;
				              python -u score.py ./tmp/result ./data/senseval3/EnglishLS.test.key ./data/senseval3/EnglishLS.sensemap"


