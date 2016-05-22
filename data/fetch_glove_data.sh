#!/bin/bash

url=http://nlp.stanford.edu/data/glove.6B.zip
fname=`basename $url`

wget $url
mkdir -p glove.6B
unzip $fname -d glove.6B/
