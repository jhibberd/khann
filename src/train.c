#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "config/digit2.h"

extern int topology[];
extern float out[LAYERS][MAX_LAYER_SIZE];
extern float err[LAYERS][MAX_LAYER_SIZE];
extern float wgt[LAYERS][MAX_LAYER_SIZE][MAX_LAYER_SIZE];
extern float trainIn[TRAIN_SIZE][NUM_INPUT_NODES];
extern float trainOut[TRAIN_SIZE][NUM_OUTPUT_NODES];

extern void save_weights_to_file(void);
extern void init_weights(void);
extern void set_outputs(float *iv);
extern void set_weights(void);
extern float error(int i);
extern void set_error_terms(int i);

static void train(void);
static void read_training_set(void);
static int classified(int i);
static void test(void);

static void test_classification_rate(void);

main () 
{
    setbuf(stdout, NULL); /* Disable stdout buffering */

    extern void load_weights_from_file(void);
    load_weights_from_file();
    read_training_set();
    test_classification_rate();
    return;

    read_training_set();
    init_weights();
    train();
    test();
    /* save_weights_to_file(); */
}

void train(void)
{
    int i, c;
    float e, *iv;
    long n = 0;

    printf("Training...\n");
    do {
        e = 0;
        c = 0; /* correctly classified */
        for (i = 0; i < TRAIN_SIZE; ++i) {
            iv = trainIn[i];
            set_outputs(iv);
            e += error(i);
            c += classified(i);
            set_error_terms(i);
            set_weights();
        }
        if (n == DEBUG_THRESHOLD) {
            printf("%f (%d/%d)\n", e, c, TRAIN_SIZE);
            n = 0;
        }
        else 
            ++n;
    } while (e > ERROR_THRESHOLD);
}

void test_classification_rate(void)
{
    int i, c;
    float *iv;
    long n = 0;

    c = 0; /* correctly classified */
    for (i = 0; i < TRAIN_SIZE; ++i) {
        iv = trainIn[i];
        set_outputs(iv);
        c += classified(i);
    }
    printf("(%d/%d)\n", c, TRAIN_SIZE);
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
        }

        ++i;
    }

    fclose(ptr_file);
}

static void test() 
{
    int i, j;
    float *iv;

    for (i = 0; i < TRAIN_SIZE; ++i) {

        for (j = 0; j < NUM_INPUT_NODES; j++)
            printf("%.0f-", trainIn[i][j]);
        printf(" -> ");

        for (j = 0; j < NUM_OUTPUT_NODES; j++)
            printf("%.0f-", trainOut[i][j]);
        printf(" -> ");

        iv = trainIn[i];
        set_outputs(iv);

        for (j = 0; j < NUM_OUTPUT_NODES; j++)
            printf("%f-", out[LAYERS-1][j]);
        printf("\n");

        }
}

