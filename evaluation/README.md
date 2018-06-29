This is the evaluation codes for the KBC task. Three algorithms are provided, including:

1. eval-rel.cpp
The algorithm takes an entity embedding file and a relation embedding file as input.
Given a query entity E and a query relation R, it ranks each candidate entity C with the score function Score(C)=-|E+R-C|, which is used in the TransE model.
There are several options to use the codes:
-train : training triplets
-test : test triplets
-entity : entity embedding file, with the embedding in the binary format.
-relation : relation embedding file, with the embedding in the binary format.
-threads : number of threads for evaluation
-k-max : hit@K
-filter : whether to filter out the training triplets during evaluation

2. eval-nodir.cpp
The algorithm only considers an entity embedding file.
For each relation, it first infers its embedding as the average direction vector between the head and tail entities with this relation. Then given a query entity E and a query relation R, it ranks each candidate entity C with the score function Score(C)=-|E+R-C|, which is used in the TransE model.
There are several options to use the codes:
-train : training triplets
-test : test triplets
-entity : entity embedding file, with the embedding in the binary format.
-threads : number of threads for evaluation
-k-max : hit@K
-filter : whether to filter out the training triplets during evaluation

3. eval-soft.cpp
The algorithm only considers an entity embedding file.
It uses a memory network to infer a unique relation embedding for each query entity and relation.
There are several options in the codes:
-train : training triplets
-test : test triplets
-entity : entity embedding file, with the embedding in the binary format.
-threads : number of threads for evaluation
-k-max : hit@K
-filter : whether to filter out the training triplets during evaluation
-k-nns: size of the memory buffer (20 is a good default)

Note:
For reading and writing with the binary embedding format, users can use the script emb-io.py in the main folder.
