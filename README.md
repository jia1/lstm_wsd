# README #

This is a work in progress state-of-the-art WSD tool.


```
#!bash

sudo docker build -t kageback/wsd . && \
sudo docker run -it --rm kageback/wsd bash -il

```

Or if you want to change the code from outside the container change the last command to:
```
#!bash
docker run -it --rm -v $(pwd):/notebooks/mycode kageback/wsd bash -il
```

Now run the experiment inside the docker.

```
#!bash

python train.py && \
python eval.py && \
python score.py ./tmp/result ./data/senseval2/Senseval2.key ./data/senseval2/sensemap
```

### Recreate COOLING 2016 workshop paper result ###
If you are interesting in recreating the results or dive into the implementation of the specifics for this paper, please checkout branch **cooling**.

### Licence ###
Distributed under [Apache v2.0 License](https://www.apache.org/licenses/LICENSE-2.0)