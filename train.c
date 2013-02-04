#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ann.h"

extern int topology[];
extern float out[LAYERS][MAX_LAYER_SIZE];
extern float err[LAYERS][MAX_LAYER_SIZE];
extern float wgt[LAYERS][MAX_LAYER_SIZE][MAX_LAYER_SIZE];
extern float trainIn[TRAIN_SIZE][NUM_INPUT_NODES];
extern float trainOut[TRAIN_SIZE][NUM_OUTPUT_NODES];

extern void save_weights_to_file(void);
extern void init_weights(void);
extern void setOutput(int t);
extern float error(int i);
extern void setErrorTerm(int i);

static void train(void);
static void read_training_set(void);
static int classified(int i);
static void test(void);

main () 
{
    read_training_set();
    init_weights();
    train();
    test();
    /* save_weights_to_file(); */
}

void train(void)
{
    int i, c;
    float e;
    long n = 0;

    printf("Learning...\n");
    fflush(stdout);
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
            fflush(stdout);
            n = 0;
        }
        else 
            ++n;
    } while (e > ERROR_THRESHOLD);
}

/* Return whether the 'i'-th element in the trainin set is correctly 
classified */
int classified(int i) 
{
    int j;
    float *o, *t;

    o = out[LAYERS-1];
    t = trainOut[i];

    j = NUM_OUTPUT_NODES;
    while (j-- > 0)
        if (*t++ != roundf(*o++))
            return 0;
    
    return 1;
}

/* Read training set from file */
static void read_training_set(void) 
{
    FILE *ptr_file;
    char buf[10000];
    char *intkn, *outtkn, *tkn;
    int i, j;

    printf("Reading training set...\n");
    fflush(stdout);
    i = 0;
    ptr_file = fopen(TRAIN_FILE, "r");
    while (fgets(buf, 10000, ptr_file) != NULL) {

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

        if (i % 1 == 0) {
            printf("\t%d/%d\n", i, TRAIN_SIZE);
            fflush(stdout);
        }

        ++i;
    }

    fclose(ptr_file);
}

static void test() 
{
    int i, j;
    for (i = 0; i < TRAIN_SIZE; ++i) {

        for (j = 0; j < NUM_INPUT_NODES; j++)
            printf("%.0f-", trainIn[i][j]);
        printf(" -> ");

        for (j = 0; j < NUM_OUTPUT_NODES; j++)
            printf("%.0f-", trainOut[i][j]);
        printf(" -> ");

        setOutput(i);

        for (j = 0; j < NUM_OUTPUT_NODES; j++)
            printf("%f-", out[LAYERS-1][j]);
        printf("\n");

        }
}

