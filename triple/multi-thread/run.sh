#!/bin/sh

./embed -entity entity.txt -relation relation.txt -triple triple.txt -output-en entity.emb -output-rl relation.emb -binary 1 -size 100 -negative 5 -samples 300 -threads 12 -alpha 0.01