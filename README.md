# README #

This is a work in progress state-of-the-art WSD tool.

docker build -t kageback/wsd .

docker run -it --rm kageback/wsd bash -il

or if you want to change the code from outside the container

docker run -it --rm -v $(pwd):/notebooks/mycode kageback/wsu bash -il

python train.py && \
python eval.py && \
python score.py ./tmp/result ./data/senseval2/Senseval2.key ./data/senseval2/sensemap

### Licence ###
Distributed under [Apache v2.0 License](https://www.apache.org/licenses/LICENSE-2.0)
