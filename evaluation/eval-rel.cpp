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

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

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
};

struct pair
{
    int id;
    real vl;
};

char train_file[MAX_STRING], test_file[MAX_STRING], entity_file[MAX_STRING], relation_file[MAX_STRING];
struct vocab_word *vocab;
int binary = 0, k_max = 1, filter = 0;
int *vocab_hash;
int vocab_max_size = 1000, vocab_size, vector_size = 0, data_size, num_threads = 1, cur_data_size = 0;
int entity_size = 0, relation_size = 0;
int train_size = 0, test_size = 0;
long long *Prank, *Qrank, *Phit, *Qhit;

BLPMatrix vec;
std::set<long long> *appear;
std::vector<triple> data;

long long hash(int h, int t)
{
    long long vl = h;
    vl = (vl << 32) + t;
    return vl;
}

int check(int h, int t, int r)
{
    return (int)(appear[r - entity_size].count(hash(h, t)));
}

real score(int h, int t, int r)
{
    return 1 - (vec.row(h) + vec.row(r) - vec.row(t)).array().abs().sum();
}

// Returns hash value of a word
int GetWordHash(char *word) {
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
    hash = hash % vocab_hash_size;
    return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
    unsigned int hash = GetWordHash(word);
    while (1) {
        if (vocab_hash[hash] == -1) return -1;
        if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
        hash = (hash + 1) % vocab_hash_size;
    }
    return -1;
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word, int id) {
    unsigned int hash;
    strcpy(vocab[id].word, word);
    hash = GetWordHash(word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = id;
    return id;
}

void ReadVector()
{
    FILE *fie, *fir;
    char ch, word[MAX_STRING];
    real f;
    
    fie = fopen(entity_file, "rb");
    fir = fopen(relation_file, "rb");
    if (fie == NULL || fir == NULL) {
        printf("Vector file not found\n");
        exit(1);
    }
    
    fscanf(fie, "%d %d", &entity_size, &vector_size);
    fscanf(fir, "%d %d", &relation_size, &vector_size);
    
    vocab_size = entity_size + relation_size;
    
    vocab = (struct vocab_word *)malloc(vocab_size * sizeof(struct vocab_word));
    vec.resize(vocab_size, vector_size);
    
    for (int k = 0; k != entity_size; k++)
    {
        fscanf(fie, "%s", word);
        ch = fgetc(fie);
        AddWordToVocab(word, k);
        for (int c = 0; c != vector_size; c++)
        {
            fread(&f, sizeof(real), 1, fie);
            //fscanf(fie, "%f", &f);
            vec(k, c) = f;
        }
    }
    
    for (int k = entity_size; k != vocab_size; k++)
    {
        fscanf(fir, "%s", word);
        ch = fgetc(fir);
        AddWordToVocab(word, k);
        for (int c = 0; c != vector_size; c++)
        {
            fread(&f, sizeof(real), 1, fir);
            //fscanf(fir, "%f", &f);
            vec(k, c) = f;
        }
    }
    
    fclose(fie);
    fclose(fir);
    
    appear = new std::set<long long>[relation_size];
    
    printf("Entity size: %d\n", entity_size);
    printf("Relation size: %d\n", relation_size);
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
            
            h = SearchVocab(sh);
            r = SearchVocab(sr);
            t = SearchVocab(st);
            
            if (h == -1 || r == -1 || t == -1) continue;
            train_size++;
            
            trip.h = h; trip.r = r; trip.t = t;
            
            appear[r - entity_size].insert(hash(h,t));
        }
        fclose(fi);
    }
    
    fi = fopen(test_file, "rb");
    while (1)
    {
        if (fscanf(fi, "%s %s %s", sh, st, sr) != 3) break;
        
        h = SearchVocab(sh);
        r = SearchVocab(sr);
        t = SearchVocab(st);
        
        if (h == -1 || r == -1 || t == -1) continue;
        test_size++;
        
        trip.h = h; trip.r = r; trip.t = t;
        
        appear[r - entity_size].insert(hash(h,t));
        
        data.push_back(trip);
    }
    fclose(fi);
    
    data_size = (int)(data.size());
    
    printf("Train size: %d\n", train_size);
    printf("Test size: %d\n", test_size);
}

void *Evaluate(void *id)
{
    long long tid = (long long)id;
    int bg = (int)(data_size / num_threads * tid);
    int ed = (int)(data_size / num_threads * (tid + 1));
    if (tid == num_threads - 1) ed = data_size;
    
    int h, r, t;
    int T = 0;
    real sc;
    
    pair *list = (pair *)malloc((k_max + 1) * sizeof(pair));
    int list_size = 0;
    
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
        
        h = data[data_id].h; r = data[data_id].r; t = data[data_id].t;
        
        sc = score(h, t, r);
        
        // use h + r to predict t
        list_size = 0;
        for (int k = 0; k != k_max + 1; k++)
        {
            list[k].id = 0;
            list[k].vl = 0;
        }
        
        crank = 1;
        for (int i = 0; i != entity_size; i++)
        {
            if (filter)
            {
                if (check(h, i, r) == 1 && i != t) continue;
            }
            
            real f = score(h, i, r);
            
            if (f > sc) crank += 1;
            
            list[list_size].id = i;
            list[list_size].vl = f;
            
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
        
        for (int k = 0; k != k_max; k++) if (list[k].id == t)
            phit += 1;
        qhit += 1;
        
        prank += crank;
        qrank += 1;
        
        // use t - r to predict h
        list_size = 0;
        for (int k = 0; k != k_max + 1; k++)
        {
            list[k].id = 0;
            list[k].vl = 0;
        }
        
        crank = 1;
        for (int i = 0; i != entity_size; i++)
        {
            if (filter)
            {
                if (check(i, t, r) == 1 && i != h) continue;
            }
            
            real f = score(i, t, r);
            
            if (f > sc) crank += 1;
            
            list[list_size].id = i;
            list[list_size].vl = f;
            
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
        
        for (int k = 0; k != list_size; k++) if (list[k].id == h)
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
    if ((i = ArgPos((char *)"-relation", argc, argv)) > 0) strcpy(relation_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-k-max", argc, argv)) > 0) k_max = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-filter", argc, argv)) > 0) filter = atoi(argv[i + 1]);
    vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
    vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
    for (long long a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    TrainModel();
    return 0;
}