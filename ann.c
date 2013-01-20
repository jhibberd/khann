/* Compile with "gcc ann.c -lm" */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* XOR
#define ERROR_THRESHOLD 0.13
#define LEARNING_RATE 0.5
#define DEBUG_THRESHOLD 1000
#define LAYERS 4
#define TRAIN_SIZE 4
#define MAX_LAYER_SIZE 4
#define TRAIN_FILE "xor.train"
#define NUM_INPUT_NODES 2
#define NUM_OUTPUT_NODES 1
int topology[] = {NUM_INPUT_NODES, 4, 4, NUM_OUTPUT_NODES};
*/

/* Binary Addition */
#define ERROR_THRESHOLD 0.13
#define LEARNING_RATE 0.5
#define DEBUG_THRESHOLD 100
#define LAYERS 4
#define TRAIN_SIZE 961
#define MAX_LAYER_SIZE 20
#define TRAIN_FILE "binadd.train"
#define NUM_INPUT_NODES 10
#define NUM_OUTPUT_NODES 6
int topology[] = {NUM_INPUT_NODES, 20, 20, NUM_OUTPUT_NODES};

int num_layers;
int num_output_nodes; 

float out[LAYERS][MAX_LAYER_SIZE];
float err[LAYERS][MAX_LAYER_SIZE];
float wgt[LAYERS][MAX_LAYER_SIZE][MAX_LAYER_SIZE];

float trainIn[TRAIN_SIZE][NUM_INPUT_NODES];
float trainOut[TRAIN_SIZE][NUM_OUTPUT_NODES];

void initWeight(void);
void readTrainingSet(void);

void setOutput(int t);
void setErrorTerm(int i);
void setWeight(void);
float error(int i);
int classified(int i);

void test(void);
float dotprod(float* a, float* b, int n);
float sigmoid(float x);

main () {

    /* Init constants */
    num_layers = sizeof(topology) / sizeof(int);
    num_output_nodes = topology[num_layers-1];

    readTrainingSet();

    initWeight();

    int i, c;
    float e;
    long n = 0;

    do {
        e = 0;
        c = 0; /* correctly classified */
        for (i = 0; i < TRAIN_SIZE; ++i) {
            setOutput(i);
            e += error(i);
            c += classified(i);
            setErrorTerm(i);
            setWeight();
        }
        if (n == DEBUG_THRESHOLD) {
            printf("%f (%d/%d)\n", e, c, TRAIN_SIZE);
            n = 0;
        }
        else 
            ++n;
    } while (e > ERROR_THRESHOLD);

    test();
}

/* Return whether the 'i'-th element in the trainin set is correctly 
classified. */
int classified(int i) {
    
    int j;
    float *o, *t;

    o = out[num_layers-1];
    t = trainOut[i];

    j = num_output_nodes;
    while (j-- > 0)
        if (*t++ != roundf(*o++))
            return 0;
    
    return 1;
}

/* Read training set from file. */
void readTrainingSet() {
    FILE *ptr_file;
    char buf[1000];
    char *intkn, *outtkn, *tkn;
    int i, j;

    i = 0;
    ptr_file = fopen(TRAIN_FILE, "r");
    while (fgets(buf, 1000, ptr_file) != NULL) {

        intkn = strtok(buf, ":");
        outtkn = strtok(NULL, ":");
    
        j = 0;
        tkn = strtok(intkn, ",");
        do {
            trainIn[i][j++] = atof(tkn);  
        } while((tkn = strtok(NULL, ",")) != NULL);

        j = 0;
        tkn = strtok(outtkn, ",");
        do {
            trainOut[i][j++] = atof(tkn);  
        } while((tkn = strtok(NULL, ",")) != NULL);

        ++i;
    }

    fclose(ptr_file);
}

void test() {
    int i, j;
    for (i = 0; i < TRAIN_SIZE; ++i) {

        for (j = 0; j < NUM_INPUT_NODES; j++)
            printf("%.0f-", trainIn[i][j]);
        printf(" -> ");

        for (j = 0; j < NUM_OUTPUT_NODES; j++)
            printf("%.0f-", trainOut[i][j]);
        printf(" -> ");

        setOutput(i);

        for (j = 0; j < num_output_nodes; j++)
            printf("%f-", out[num_layers-1][j]);
        printf("\n");
        }
}

void initWeight() {
    int l, j, k;

    /* Seed random number generator */
    srand(time(NULL));
    /* srand(10); Deterministic */

    /* Assign each weight a number between -0.5 and +0.5 */
    for (l = 1; l < num_layers; ++l)
        for (j = 0; j < topology[l]; ++j)
            for (k = 0; k < topology[l-1]; ++k)
                wgt[l][j][k] = ((double)rand() / (double)RAND_MAX) - 0.5;
}


/* Sets the output value of each node according to the current network weights
 * and the input vector ('iv') belonging to the 't'th element in the training
 * set. */
void setOutput(int t) {
    int j, l;
    float o;
    float *iv;
    
    iv = trainIn[t];
    for (j = 0; j < topology[0]; ++j)
        out[0][j] = *iv++;

    for (l = 1; l < num_layers; ++l) {
        for (j = 0; j < topology[l]; ++j) {
            o = dotprod(wgt[l][j], out[l-1], topology[l-1]);
            out[l][j] = sigmoid(o);
        } 
    }
}

void setErrorTerm(int i) {
    int n;
    float *o, *t, *e;

    /* error term for output nodes */
    n = num_output_nodes;
    e = err[num_layers-1];
    o = out[num_layers-1];
    t = trainOut[i];
    while (n-- > 0) 
        *e++ = *o * (1.0 - *o) * (*t++ - *o++);

    int l, j;
    float er;
    /* set error terms for hidden nodes */
    for (l = num_layers-2; l >= 0; --l) {
        o = out[l];
        e = err[l];
        for (j = 0; j < topology[l]; j++) {

            int k;
            float ws[topology[l+1]];
            for (k = 0; k < topology[l+1]; ++k)
                ws[k] = wgt[l+1][k][j];

            er = dotprod(ws, err[l+1], topology[l+1]);
            *e++ = *o * (1.0 - *o) * er;
            o++;
        }
     }
}

void setWeight(void) {
    int l, i, n;
    float *w, *o, f;

    for (l = 1; l < num_layers; ++l)
        for (i = 0; i < topology[l]; ++i) {
            w = wgt[l][i];
            o = out[l-1];
            f = LEARNING_RATE * err[l][i];
            n = topology[l-1];
            while (n-- > 0)
                *w++ += f * *o++;
        }
}

float error(int i) {

    int j;
    float e, *o, *t;

    o = out[num_layers-1];
    t = trainOut[i];

    e = 0, j = num_output_nodes;
    while (j-- > 0)
        e += pow((*t++ - *o++), 2);

    return e * 0.5;
}

/* Compute the dot product of two n length arrays. */
float dotprod(float* a, float* b, int n) {
    float s = 0;
    while (n-- > 0)
        s += *a++ * *b++;
    return s;
}

float sigmoid(float x) {
    return 1 / (1 + expf(-x));
}

