/* Compile with "gcc ann.c -lm" */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "config/xor.h"

/* XOR */
int topology[] = {NUM_INPUT_NODES, 4, 4, NUM_OUTPUT_NODES};

/* Binary Addition
int topology[] = {NUM_INPUT_NODES, 20, 20, NUM_OUTPUT_NODES};
*/

/* Digit Recogniser
int topology[] = {NUM_INPUT_NODES, 1568, 784, NUM_OUTPUT_NODES};
*/

/* Digit Recogniser (compressed x2)
int topology[] = {NUM_INPUT_NODES, 392, 196, NUM_OUTPUT_NODES};
*/

float out[LAYERS][MAX_LAYER_SIZE];
float err[LAYERS][MAX_LAYER_SIZE];
float wgt[LAYERS][MAX_LAYER_SIZE][MAX_LAYER_SIZE];
float trainIn[TRAIN_SIZE][NUM_INPUT_NODES];
float trainOut[TRAIN_SIZE][NUM_OUTPUT_NODES];

extern void save_weights_to_file(void);
extern void load_weights_from_file(void);

void init_weights(void);
void setOutput(int t);
void setErrorTerm(int i);
void setWeight(void);
float error(int i);
float *eval(float *iv);
static float dotprod(float* a, float* b, int n);
static float sigmoid(float x);

float *eval(float *iv) 
{
    static short loaded = 0;
    if (!loaded) {
        load_weights_from_file();
        loaded = 1;
        printf("%s", "loading");
    }

    int j, l;
    float o;
    
    for (j = 0; j < topology[0]; ++j)
        out[0][j] = *iv++;

    for (l = 1; l < LAYERS; ++l) {
        for (j = 0; j < topology[l]; ++j) {
            o = dotprod(wgt[l][j], out[l-1], topology[l-1]);
            out[l][j] = sigmoid(o);
        } 
    }

    return out[LAYERS-1];
}

void init_weights(void) 
{
    int l, j, k;

    /* Assign each weight a number between -0.5 and +0.5 */
    printf("Initialising weights...\n");
    fflush(stdout);
    srand(time(NULL));
    for (l = 1; l < LAYERS; ++l)
        for (j = 0; j < topology[l]; ++j)
            for (k = 0; k < topology[l-1]; ++k)
                wgt[l][j][k] = ((double)rand() / (double)RAND_MAX) - 0.5;
}


/* Sets the output value of each node according to the current network weights
 * and the input vector ('iv') belonging to the 't'th element in the training
 * set. */
void setOutput(int t) 
{
    int j, l;
    float o;
    float *iv;
    
    iv = trainIn[t];
    for (j = 0; j < topology[0]; ++j)
        out[0][j] = *iv++;

    for (l = 1; l < LAYERS; ++l) {
        for (j = 0; j < topology[l]; ++j) {
            o = dotprod(wgt[l][j], out[l-1], topology[l-1]);
            out[l][j] = sigmoid(o);
        } 
    }
}

void setErrorTerm(int i) 
{
    int n;
    float *o, *t, *e;

    /* error term for output nodes */
    n = NUM_OUTPUT_NODES;
    e = err[LAYERS-1];
    o = out[LAYERS-1];
    t = trainOut[i];
    while (n-- > 0) { 
        *e++ = *o * (1.0 - *o) * (*t++ - *o);
        o++;
        }

    int l, j;
    float er;
    /* set error terms for hidden nodes */
    for (l = LAYERS-2; l >= 0; --l) {
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

void setWeight(void) 
{
    int l, i, n;
    float *w, *o, f;

    for (l = 1; l < LAYERS; ++l)
        for (i = 0; i < topology[l]; ++i) {
            w = wgt[l][i];
            o = out[l-1];
            f = LEARNING_RATE * err[l][i];
            n = topology[l-1];
            while (n-- > 0)
                *w++ += f * *o++;
        }
}

float error(int i) 
{

    int j;
    float e, *o, *t;

    o = out[LAYERS-1];
    t = trainOut[i];

    e = 0, j = NUM_OUTPUT_NODES;
    while (j-- > 0)
        e += pow((*t++ - *o++), 2);

    return e * 0.5;
}

/* Compute the dot product of two n length arrays. */
static float dotprod(float* a, float* b, int n) 
{
    float s = 0;
    while (n-- > 0)
        s += *a++ * *b++;
    return s;
}

static float sigmoid(float x) 
{
    return 1 / (1 + expf(-x));
}

