# image for running tensorflow on titan using the GPU
FROM gcr.io/tensorflow/tensorflow:0.7.0
MAINTAINER Mikael Kågebäck <kageback@chalmers.se>
	

RUN apt-get update && apt-get install -y \
	nano \
	python-lxml \
	curl \
	unzip

RUN pip install sklearn nltk lxml

WORKDIR /src

# Create data and tmp dir
RUN mkdir tmp && cd tmp && mkdir model && cd model && mkdir 2 && cd ../..

#Download Glove
#RUN mkdir data && cd data && \
#      curl -O http://nlp.stanford.edu/data/glove.6B.zip && \
#      unzip glove.6B.zip -d ./glove.6B
#Get models
#RUN cp /home/camille/2/
### Download Senseval
#RUN cd data && mkdir senseval2 && cd senseval2 && \
#	curl http://www.d.umn.edu/~tpederse/Data/lexical-sample.dtd.tar.gz | tar -xz && \
#	curl http://www.d.umn.edu/~tpederse/Data/Sval2.xml.tar.gz | tar -xz && \
#	curl http://www.d.umn.edu/~tpederse/Data/Sval2.keys.tar.gz | tar -xz && \
#        #curl http://www.hipposmond.com/senseval2/Results/senseval2-corpora.tgz | tar -xz && \
#      cd .. && \
#      mkdir senseval3 && cd senseval3 && \
#        curl http://web.eecs.umich.edu/~mihalcea/senseval/senseval3/data/EnglishLS/EnglishLS.train.tar.gz | tar -xz && \
#	curl http://web.eecs.umich.edu/~mihalcea/senseval/senseval3/data/EnglishLS/EnglishLS.test.tar.gz | tar -xz

# Add files in the path of the Docker-file to the working directory of the container
copy . .

