# TransE
This is an implementation of the TransE model.
The TransE model uses some triplets as training data to learn entity and relation embeddings.

The codes rely on two external packages (Eigen and GSL). After installing the packages, users need to change the package paths in the makefile. Then we can compile the code and use the running script run.sh to train.

Options:
-entity : entity vocabulary file, which consists of N lines, where N is the number of entities. Each line contains an entity name.
-relation : relation vocabulary file, which consists of R lines, where R is the number of relations. Each line contains a relation name.
-triple : training triplet file. Each line describes a triplet, with the format <Head> <Tail> <Relation>
-output-en : output entity embedding file
-output-rl : output relation embedding file
-binary : whether to output embeddings in the binary format
-size : embedding dimension
-samples : number of training samples (in million), 300 is a good default.
-alpha : learning rate. 0.001 is a good default.
-threads : number of threads for training
