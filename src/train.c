#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "config/binadd.h"
#include "ann.h"


static struct training_set load_training_set(void);
static void free_training_set(struct training_set *t);

extern int topology[];
extern float out[LAYERS][MAX_LAYER_SIZE];
extern float err[LAYERS][MAX_LAYER_SIZE];
extern float wgt[LAYERS][MAX_LAYER_SIZE][MAX_LAYER_SIZE];

extern void save_weights_to_file(void);
extern void init_weights(void);
extern void set_outputs(float *iv);
extern void set_weights(void);
extern float error(int i, struct training_set *t);
extern void set_error_terms(int i, struct training_set *t);

static void train(struct training_set *t);
static int classified(int i, struct training_set *t);
static void test(struct training_set *t);

static void test_classification_rate(void);

main () 
{
    struct training_set t;

    setbuf(stdout, NULL); /* Disable stdout buffering */
    t = load_training_set();
    init_weights();
    train(&t);
    test(&t);
    /* save_weights_to_file(); */
    free_training_set(&t);
}

void train(struct training_set *t)
{
    int i, c;
    float e;
    long n = 0;

    printf("Training...\n");
    do {
        e = 0;
        c = 0; /* correctly classified */
        for (i = 0; i < t->n; ++i) {
            set_outputs(t->iv[i]);
            e += error(i, t);
            c += classified(i, t);
            set_error_terms(i, t);
            set_weights();
        }
        if (n == DEBUG_THRESHOLD) {
            printf("%f (%d/%d)\n", e, c, t->n);
            n = 0;
        }
        else 
            ++n;
    } while (e > ERROR_THRESHOLD);
}

/* Return whether the 'i'-th element in the trainin set is correctly 
classified */
int classified(int i, struct training_set *t) 
{
    int j;
    float *o, *tgt;

    o = out[LAYERS-1];
    tgt = t->ov[i];

    j = t->size_ov;
    while (j-- > 0)
        if (*tgt++ != roundf(*o++))
            return 0;
    
    return 1;
}

static void test(struct training_set *t) 
{
    int i, j;

    for (i = 0; i < t->n; ++i) {

        for (j = 0; j < t->size_iv; j++)
            printf("%.0f-", t->iv[i][j]);
        printf(" -> ");

        for (j = 0; j < t->size_ov; j++)
            printf("%.0f-", t->ov[i][j]);
        printf(" -> ");

        set_outputs(t->iv[i]);

        for (j = 0; j < t->size_ov; j++)
            printf("%f-", out[LAYERS-1][j]);
        printf("\n");

        }
}

static struct training_set load_training_set(void)
{
    struct training_set t;

    /* Make an unbuffered pass through the training set file to count the 
     * number of elements */
    t.n = 0;
    int ch;
    FILE *fp = fopen(TRAIN_FILE, "r");
    while (EOF != (ch = fgetc(fp)))
        if (ch == '\n')
            ++t.n; 

    /* Make a pass through the first element to count the size of each input
     * and output vector */
    int *n;
    rewind(fp);
    t.size_iv = 0;
    t.size_ov = 0;
    n = &t.size_iv;
    while ('\n' != (ch = fgetc(fp)))
        if (ch == ',')
            ++*n;
        else if (ch == ':') {
            ++*n;
            n = &t.size_ov;
        }
    ++t.size_ov;

    /* Allocate enough memory for an array to hold all input and output vectors
     * of the training set */
    int i;
    t.iv = malloc(t.n * sizeof(float *));
    t.ov = malloc(t.n * sizeof(float *));
    for (i = 0; i < t.n; i++) {
        t.iv[i] = malloc(t.size_iv * sizeof(float));
        t.ov[i] = malloc(t.size_ov * sizeof(float));
    }    

    /* Make a second pass through the training set file and load it into both
     * arrays */
    char v[DBL_DIG+2]; /* Largest string representation of a float */
    div_t d;
    int line_size;
    rewind(fp);
    i = 0;
    line_size = t.size_iv + t.size_ov;
    while (fscanf(fp, "%[^,:\n]%*c", v) != EOF) {
        d = div(i, line_size);
        if (d.rem < t.size_iv)
            t.iv[d.quot][d.rem] = (float) atof(v);
        else
            t.ov[d.quot][d.rem - t.size_iv] = (float) atof(v);
        ++i;
    }
    fclose(fp);

    return t;
}

/* Free memory dynamically allocated for the training set */
static void free_training_set(struct training_set *t)
{
    int i;
    for (i = 0; i < t->n; i++) {
        free(t->iv[i]);
        free(t->ov[i]);
    }
    free(t->iv);
    free(t->ov);
}

