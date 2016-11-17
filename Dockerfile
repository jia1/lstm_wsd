# image for running tensorflow on titan using the GPU
FROM gcr.io/tensorflow/tensorflow:0.7.0
MAINTAINER Mikael Kågebäck <kageback@chalmers.se>
	

RUN apt-get update && apt-get install -y \
	nano \
	python-lxml

RUN pip install sklearn nltk lxml

# Add files in the path of the Dockerfile to the working directory of the container
copy . ./mycode

RUN mkdir tmp && cd tmp && mkdir model && cd model && mkdir 2 && cd ../..

WORKDIR ./mycode

#############################################################################################################
# Here you can add all kinds of stuff for building your image, e.g install programs or download datasets.
# #install packages
# RUN apt-get update && apt-get install -y \	
#	curl \
#	tar	
#
# #fetch dataset
# RUN 	curl http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz | tar -xz
#
# RUN	mkdir /data && \
#	mv simple-examples /data/simple-examples
#
# create volume for data. Not nessasary but will make accessing faster.
# VOLUME ["/data"]
#
# #Change working directory perhaps
# WORKDIR <some dir>

# #change default command 
# CMD <whathever you would like to run>
#############################################################################################################
