//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <set>
#include <map>
#include <Eigen/Dense>
#include <iostream>

#define MAX_STRING 1000
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

typedef Eigen::Matrix< real, Eigen::Dynamic,
Eigen::Dynamic, Eigen::RowMajor | Eigen::AutoAlign >
BLPMatrix;

typedef Eigen::Matrix< real, 1, Eigen::Dynamic,
Eigen::RowMajor | Eigen::AutoAlign >
BLPVector;

struct vocab_word
{
    char word[MAX_STRING];
};

struct triple
{
    int h, r, t;
    friend bool operator < (triple t1, triple t2)
    {
        if (t1.h == t2.h)
        {
            if (t1.r == t2.r) return t1.t < t2.t;
            return t1.r < t2.r;
        }
        return t1.h < t2.h;
    }
};

struct pair
{
    int id;
    real vl;
};

struct kmax_list
{
    pair *list;
    int k_max, list_size;
    
    void init(int k)
    {
        k_max = k;
        list = (pair *)malloc((k_max + 1) * sizeof(pair));
        list_size = 0;
        for (int k = 0; k != k_max + 1; k++)
        {
            list[k].id = -1;
            list[k].vl = -1;
        }
    }
    
    void clear()
    {
        list_size = 0;
        for (int k = 0; k != k_max + 1; k++)
        {
            list[k].id = -1;
            list[k].vl = -1;
        }
    }
    
    void add(pair pr)
    {
        list[list_size].id = pr.id;
        list[list_size].vl = pr.vl;
        
        for (int k = list_size - 1; k >= 0; k--)
        {
            if (list[k].vl < list[k + 1].vl)
            {
                int tmp_id = list[k].id;
                real tmp_vl = list[k].vl;
                list[k].id = list[k + 1].id;
                list[k].vl = list[k + 1].vl;
                list[k + 1].id = tmp_id;
                list[k + 1].vl = tmp_vl;
            }
            else
                break;
        }
        
        if (list_size < k_max) list_size++;
    }
};

struct relation2data
{
    std::vector<triple> data;
    BLPMatrix vector;
    int data_size, vector_size;
    
    void set(std::vector<triple> triples, BLPMatrix directions)
    {
        data = triples;
        vector = directions;
        
        vector_size = (int)(vector.cols());
        data_size = (int)(data.size());
    }
};

char train_file[MAX_STRING], test_file[MAX_STRING], entity_file[MAX_STRING];
struct vocab_word *entity, *relation;
int binary = 0, k_max = 1, k_nns = 5, filter = 0;
int *entity_hash, *relation_hash;
int vector_size = 0, data_size, num_threads = 1, cur_data_size = 0;
int entity_size = 0, relation_size = 0, relation_max_size = 1000;
int train_size = 0, test_size = 0;
long long *Prank, *Qrank, *Phit, *Qhit;
relation2data *rlt2data;

BLPMatrix vec;
std::set<triple> appear;
std::vector<triple> train_data, test_data;

long long hash(int h, int t)
{
    long long vl = h;
    vl = (vl << 32) + t;
    return vl;
}

int check(int h, int t, int r)
{
    triple trip;
    trip.h = h; trip.t = t; trip.r = r;
    return (int)(appear.count(trip));
}

real score(int h, int t, BLPVector &dir, char pst)
{
    //return 1 - (vec.row(h) + dir - vec.row(t)).array().abs().sum();
    if (pst == 'h') return (vec.row(t) - dir) * vec.row(h).transpose();
    else if (pst == 't') return (vec.row(h) + dir) * vec.row(t).transpose();
    return 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
    hash = hash % hash_size;
    return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchEntity(char *word) {
    unsigned int hash = GetWordHash(word);
    while (1) {
        if (entity_hash[hash] == -1) return -1;
        if (!strcmp(word, entity[entity_hash[hash]].word)) return entity_hash[hash];
        hash = (hash + 1) % hash_size;
    }
    return -1;
}

// Adds a word to the vocabulary
int AddWordToEntity(char *word, int id) {
    unsigned int hash;
    strcpy(entity[id].word, word);
    hash = GetWordHash(word);
    while (entity_hash[hash] != -1) hash = (hash + 1) % hash_size;
    entity_hash[hash] = id;
    return id;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchRelation(char *word) {
    unsigned int hash = GetWordHash(word);
    while (1) {
        if (relation_hash[hash] == -1) return -1;
        if (!strcmp(word, relation[relation_hash[hash]].word)) return relation_hash[hash];
        hash = (hash + 1) % hash_size;
    }
    return -1;
}

// Adds a word to the vocabulary
int AddWordToRelation(char *word) {
    unsigned int hash, length = strlen(word) + 1;
    strcpy(relation[relation_size].word, word);
    relation_size++;
    // Reallocate memory if needed
    if (relation_size + 2 >= relation_max_size) {
        relation_max_size += 1000;
        relation = (struct vocab_word *)realloc(relation, relation_max_size * sizeof(struct vocab_word));
    }
    hash = GetWordHash(word);
    while (relation_hash[hash] != -1) hash = (hash + 1) % hash_size;
    relation_hash[hash] = relation_size - 1;
    return relation_size - 1;
}

void ReadVector()
{
    FILE *fie;
    char ch, word[MAX_STRING];
    real f;
    
    fie = fopen(entity_file, "rb");
    if (fie == NULL) {
        printf("Vector file not found\n");
        exit(1);
    }
    
    fscanf(fie, "%d %d", &entity_size, &vector_size);
    
    entity = (struct vocab_word *)malloc(entity_size * sizeof(struct vocab_word));
    vec.resize(entity_size, vector_size);
    
    for (int k = 0; k != entity_size; k++)
    {
        fscanf(fie, "%s", word);
        ch = fgetc(fie);
        AddWordToEntity(word, k);
        for (int c = 0; c != vector_size; c++)
        {
            fread(&f, sizeof(real), 1, fie);
            //fscanf(fie, "%f", &f);
            vec(k, c) = f;
        }
        vec.row(k) /= vec.row(k).norm();
    }
    
    fclose(fie);
    
    printf("Entity size: %d\n", entity_size);
    printf("Vector size: %d\n", vector_size);
}

void ReadTriple()
{
    FILE *fi;
    char sh[MAX_STRING], st[MAX_STRING], sr[MAX_STRING];
    int h, t, r;
    triple trip;
    
    if (train_file[0] != 0)
    {
        fi = fopen(train_file, "rb");
        while (1)
        {
            if (fscanf(fi, "%s %s %s", sh, st, sr) != 3) break;
            
            h = SearchEntity(sh);
            t = SearchEntity(st);
            
            if (h == -1 || t == -1) continue;
            
            r = SearchRelation(sr);
            if (r == -1) r = AddWordToRelation(sr);
            
            train_size++;
            
            trip.h = h; trip.r = r; trip.t = t;
            
            appear.insert(trip);
            
            train_data.push_back(trip);
        }
        fclose(fi);
    }
    
    fi = fopen(test_file, "rb");
    while (1)
    {
        if (fscanf(fi, "%s %s %s", sh, st, sr) != 3) break;
        
        h = SearchEntity(sh);
        r = SearchRelation(sr);
        t = SearchEntity(st);
        
        if (h == -1 || r == -1 || t == -1) continue;
        test_size++;
        
        trip.h = h; trip.r = r; trip.t = t;
        
        appear.insert(trip);
        
        test_data.push_back(trip);
    }
    fclose(fi);
    
    data_size = (int)(test_data.size());
    
    printf("Relation size: %d\n", relation_size);
    printf("Train size: %d\n", train_size);
    printf("Test size: %d\n", test_size);
}

void Process()
{
    rlt2data = new relation2data [relation_size];
    std::vector<triple> triples;
    BLPMatrix directions;
    for (int rr = 0; rr != relation_size; rr++)
    {
        triples.clear();
        for (int k = 0; k != train_size; k++)
        {
            int r = train_data[k].r;
            if (r != rr) continue;
            triples.push_back(train_data[k]);
        }
        
        int size = (int)(triples.size());
        
        directions.resize(size, vector_size);
        
        for (int k = 0; k != size; k++)
        {
            int h = triples[k].h;
            int t = triples[k].t;
            directions.row(k) = vec.row(t) - vec.row(h);
        }
        
        rlt2data[rr].set(triples, directions);
    }
}

void *Evaluate(void *id)
{
    long long tid = (long long)id;
    int bg = (int)(data_size / num_threads * tid);
    int ed = (int)(data_size / num_threads * (tid + 1));
    if (tid == num_threads - 1) ed = data_size;
    
    int h, r, t;
    int T = 0;
    double sum = 0;
    real sc;
    pair pr;
    BLPVector dir;
    dir.resize(vector_size);
    
    kmax_list nblist, rklist;
    nblist.init(k_nns);
    rklist.init(k_max);
    
    long long prank = 0, qrank = 0, phit = 0, qhit = 0;
    long long crank = 0;
    
    for (int data_id = bg; data_id != ed; data_id++)
    {
        T++;
        if (T % 10 == 0)
        {
            cur_data_size += 10;
            printf("%cProgress: %.2f%%", 13, 100.0 * cur_data_size / data_size);
            fflush(stdout);
        }
        
        h = test_data[data_id].h; r = test_data[data_id].r; t = test_data[data_id].t;
        
        // use h + r to predict t
        nblist.clear();
        for (int k = 0; k != rlt2data[r].data_size; k++)
        {
            int hh = rlt2data[r].data[k].h;
            real f = vec.row(h) * vec.row(hh).transpose();
            pr.id = k; pr.vl = f;
            nblist.add(pr);
        }
        
        dir.setZero();
        sum = 0;
        for (int k = 0; k != k_nns; k++)
        {
            int id = nblist.list[k].id;
            if (id == -1) continue;
            dir += rlt2data[r].vector.row(id);
            sum += 1;
        }
        if (sum != 0) dir /= sum;
        
        sc = score(h, t, dir, 't');
        
        rklist.clear();
        
        crank = 1;
        for (int i = 0; i != entity_size; i++)
        {
            if (filter)
            {
                if (check(h, i, r) == 1 && i != t) continue;
            }
            
            real f = score(h, i, dir, 't');
            
            if (f > sc) crank += 1;
            
            pr.id = i;
            pr.vl = f;
            rklist.add(pr);
        }
        
        for (int k = 0; k != k_max; k++) if (rklist.list[k].id == t)
            phit += 1;
        qhit += 1;
        
        prank += crank;
        qrank += 1;
        
        
        
        // use t - r to predict h
        nblist.clear();
        for (int k = 0; k != rlt2data[r].data_size; k++)
        {
            int tt = rlt2data[r].data[k].t;
            real f = vec.row(t) * vec.row(tt).transpose();
            pr.id = k; pr.vl = f;
            nblist.add(pr);
        }
        
        dir.setZero();
        sum = 0;
        for (int k = 0; k != k_nns; k++)
        {
            int id = nblist.list[k].id;
            if (id == -1) continue;
            dir += rlt2data[r].vector.row(id);
            sum += 1;
        }
        if (sum != 0) dir /= sum;
        
        sc = score(h, t, dir, 'h');
        
        rklist.clear();
        
        crank = 1;
        for (int i = 0; i != entity_size; i++)
        {
            if (filter)
            {
                if (check(i, t, r) == 1 && i != h) continue;
            }
            
            real f = score(i, t, dir, 'h');
            
            if (f > sc) crank += 1;
            
            pr.id = i;
            pr.vl = f;
            rklist.add(pr);
        }
        
        for (int k = 0; k != k_max; k++) if (rklist.list[k].id == h)
            phit += 1;
        qhit += 1;
        
        prank += crank;
        qrank += 1;
    }
    Phit[tid] = phit;
    Qhit[tid] = qhit;
    Prank[tid] = prank;
    Qrank[tid] = qrank;
    pthread_exit(NULL);
}

void TrainModel()
{
    long a;
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    long long sPrank = 0, sQrank = 0, sPhit = 0, sQhit = 0;
    
    Prank = (long long *)calloc(num_threads, sizeof(long long));
    Qrank = (long long *)calloc(num_threads, sizeof(long long));
    Phit = (long long *)calloc(num_threads, sizeof(long long));
    Qhit = (long long *)calloc(num_threads, sizeof(long long));
    
    ReadVector();
    ReadTriple();
    Process();
    
    if (k_nns == 0) k_nns = train_size;
    
    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, Evaluate, (void *)a);
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    printf("\n");
    
    for (a = 0; a != num_threads; a++)
    {
        sPrank += Prank[a];
        sQrank += Qrank[a];
        
        sPhit += Phit[a];
        sQhit += Qhit[a];
    }
    
    printf("Hit@%d: %.2lf%% Rank: %.2lf\n", k_max, 100 * (double)(sPhit) / (double)(sQhit), (double)(sPrank) / (double)(sQrank));
}

int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}

int main(int argc, char **argv) {
    int i;
    if (argc == 1) {
        printf("WORD VECTOR estimation toolkit v 0.1b\n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-train <file>\n");
        printf("\t\tUse text data from <file> to train the model\n");
        printf("\t-output <file>\n");
        printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
        printf("\t-size <int>\n");
        printf("\t\tSet size of word vectors; default is 100\n");
        printf("\nExamples:\n");
        printf("./btm2vec -train btm.txt -output vec.txt -debug 2 -size 200 -samples 100 -negative 5 -hs 0 -binary 1\n\n");
        return 0;
    }
    if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-test", argc, argv)) > 0) strcpy(test_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-entity", argc, argv)) > 0) strcpy(entity_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-k-max", argc, argv)) > 0) k_max = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-k-nns", argc, argv)) > 0) k_nns = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-filter", argc, argv)) > 0) filter = atoi(argv[i + 1]);
    entity_hash = (int *)calloc(hash_size, sizeof(int));
    for (long long a = 0; a < hash_size; a++) entity_hash[a] = -1;
    relation = (struct vocab_word *)calloc(relation_max_size, sizeof(struct vocab_word));
    relation_hash = (int *)calloc(hash_size, sizeof(int));
    for (long long a = 0; a < hash_size; a++) relation_hash[a] = -1;
    TrainModel();
    return 0;
}
