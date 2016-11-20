#!/bin/bash

docker build -t kageback/wsd . && \
docker run -t --rm kageback/wsd /bin/bash -c "python -u train.py 3;
					      python -u eval.py 3;
				              python -u score.py ./tmp/result ./data/senseval3/EnglishLS.test.key ./data/senseval3/EnglishLS.sensemap"


