#!/bin/sh

./data2w -text text.txt -output-ww network.txt -output-word entity.txt -window 5 -min-count 10

./embed -entity entity.txt -relation relation.txt -network network.txt -triple triple.txt -output-en entity.emb -output-rl relation.emb -binary 1 -size 100 -negative 5 -samples 300 -threads 12 -alpha 0.01