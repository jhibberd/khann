#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "config/binadd.h"
#include "ann.h"

#define sigmoid(x) 1 / (1 + expf(-(x)))
#define LEARNING_RATE 0.5

static struct training_set load_training_set(void);
static void free_training_set(struct training_set *t);
static struct network mknetwork(void); 
static void free_network(struct network *n);

const int topology[] = TOPOLOGY;

/* void save_weights_to_file(void); */
void set_outputs(struct network *n, float *iv);
void set_weights(struct network *n);
float error(struct network *n, struct training_set *t, int i);
void set_error_terms(struct network *n, struct training_set *t, int ti);

static void train(struct training_set *t, struct network *n);
static int classified(struct network *n, struct training_set *t, int i);
static void test(struct training_set *t, struct network *n);

main () 
{
    struct training_set t;
    struct network n;

    setbuf(stdout, NULL);
    t = load_training_set();
    n = mknetwork();
    train(&t, &n);
    test(&t, &n);
    /* save_weights_to_file(); */
    free_training_set(&t);
    free_network(&n);
}

static struct network mknetwork(void) 
{
    struct network n;
    int i, max_layer_size;

    /* Count layers in network */
    n.layers = sizeof(topology) / sizeof(int);

    /* Get size of largest layer (all arrays will be 'square' for now) */
    max_layer_size = 0;
    for (i = 0; i < n.layers; i++)
        if (topology[i] > max_layer_size)
            max_layer_size = topology[i];

    /* Allocate memory for all arrays */
    n.output =  mkarr2d(n.layers, max_layer_size); 
    n.error =   mkarr2d(n.layers, max_layer_size); 
    n.weight =  mkarr3d(n.layers, max_layer_size, max_layer_size);

    /* Assign random weights (between -0.5 and +0.5) to the network */
    int l, j, k;
    float w;
    printf("Initialising weights...\n");
    srand(time(NULL));
    for (l = 1; l < n.layers; ++l)
        for (j = 0; j < topology[l]; ++j)
            for (k = 0; k < topology[l-1]; ++k) {
                w = ((double)rand() / (double)RAND_MAX) - 0.5;
                *getarr3d(&n.weight, l, j, k) = w;
            }
    
    return n;
}

/* TODO(jhibberd) Topology to be a property of the network struct */
void set_outputs(struct network *n, float *iv) 
{
    int i, j;

    /* Set first layer of output nodes to value of input vector */ 
    for (i = 0; i < topology[0]; ++i)
        *getarr2d(&n->output, 0, i) = *iv++;

    /* Recursively set the output values of all downstream nodes, using the
     * current weight values */
    for (i = 1; i < n->layers; ++i)
        for (j = 0; j < topology[i]; ++j) {

            float dp = 0, *pw, *po;
            int c;

            pw = getarr3d(&n->weight, i, j, 0);
            po = getarr2d(&n->output, i-1, 0);
            c = topology[i-1];
            while (c-- > 0)
                dp += *pw++ * *po++;

            *getarr2d(&n->output, i, j) = sigmoid(dp);
        } 
}

/* TODO(jhibberd) Add a doc string to each function */
/* TODO(jhibberd) Mark all functions as 'static' that are internal to this
 * translation unit */
void set_error_terms(struct network *n, struct training_set *t, int ti) 
{
    int i, j;
    float *o, *e;

    /* Set the error terms (e) for the final output nodes (o), based on the 
     * expected values (z) in the training set (t) */
    float *z;
    i = t->size_ov;
    e = getarr2d(&n->error, n->layers -1, 0);
    o = getarr2d(&n->output, n->layers -1, 0);
    z = t->ov[ti];
    while (i-- > 0) { 
        *e++ = *o * (1.0 - *o) * (*z++ - *o);
        ++o;
        }

    /* Propagate the error term backwards through the network. This can be
     * loosely interpreted as distributing the error across all nodes based
     * on how "responsible" each node is for the error. */
    for (i = n->layers -2; i >= 0; --i) {
        o = getarr2d(&n->output, i, 0);
        e = getarr2d(&n->error, i, 0);
        for (j = 0; j < topology[i]; ++j) {

            int k;
            float dp = 0, w, *e2;
            
            e2 = getarr2d(&n->error, i+1, 0);
            for (k = 0; k < topology[i+1]; ++k) {
                w = *getarr3d(&n->weight, i+1, k, j);
                dp += w * *e2++;
            }

            *e++ = *o * (1.0 - *o) * dp;
            ++o;
        }
    }
}

void set_weights(struct network *n) 
{
    int l, i, x;
    float *w, *o, f;

    for (l = 1; l < n->layers; ++l)
        for (i = 0; i < topology[l]; ++i) {
            w = getarr3d(&n->weight, l, i, 0);
            o = getarr2d(&n->output, l-1, 0); 
            f = LEARNING_RATE * *getarr2d(&n->error, l, i);
            x = topology[l-1];
            while (x-- > 0)
                *w++ += f * *o++;
        }
}

/* TODO(jhibberd) Better function name */
float error(struct network *n, struct training_set *t, int i) 
{
    int j;
    float e, *o, *tgt;

    o = getarr2d(&n->output, n->layers -1, 0);
    tgt = t->ov[i];

    e = 0, j = t->size_ov;
    while (j-- > 0)
        e += pow((*tgt++ - *o++), 2);

    return e * 0.5;
}

void train(struct training_set *t, struct network *n)
{
    int i, c;
    float e;
    long x = 0;

    /* TODO(jhibberd) Only call classified if correct iteration */
    /* TODO(jhibberd) More verbose variable names */
    printf("Training...\n");
    do {
        e = 0;
        c = 0; /* correctly classified */
        for (i = 0; i < t->n; ++i) {
            set_outputs(n, t->iv[i]);
            e += error(n, t, i);
            c += classified(n, t, i);
            set_error_terms(n, t, i);
            set_weights(n);
        }
        if (x == DEBUG_THRESHOLD) {
            printf("%f (%d/%d)\n", e, c, t->n);
            x = 0;
        }
        else 
            ++x;
    } while (e > ERROR_THRESHOLD);
}

int classified(struct network *n, struct training_set *t, int i) 
{
    int j;
    float *o, *tgt;

    o = getarr2d(&n->output, n->layers -1, 0);
    tgt = t->ov[i];

    j = t->size_ov;
    while (j-- > 0)
        if (*tgt++ != roundf(*o++))
            return 0;
    
    return 1;
}

/* TODO(jhibberd) Better function name */
static void test(struct training_set *t, struct network *n) 
{
    int i, j;
    float x;

    for (i = 0; i < t->n; ++i) {

        for (j = 0; j < t->size_iv; j++)
            printf("%.0f-", t->iv[i][j]);
        printf(" -> ");

        for (j = 0; j < t->size_ov; j++)
            printf("%.0f-", t->ov[i][j]);
        printf(" -> ");

        set_outputs(n, t->iv[i]);

        for (j = 0; j < t->size_ov; j++) {
            x = *getarr2d(&n->output, n->layers -1, j);
            printf("%f-", x);
        }
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

/* Free memory dynamically allocated for the network */
static void free_network(struct network *n)
{
    free(n->output.arr);
    free(n->error.arr);
    free(n->weight.arr);
}

/* Construct (and allocate memory for) a 3D array */
struct arr3d mkarr3d(int x, int y, int z)
{
    struct arr3d a;
    a.arr = malloc(x * y * z * sizeof(float));
    a.dx = x;
    a.dy = y;
    a.dz = z;
    return a;
}

/* Construct (and allocate memory for) a 2D array */
struct arr2d mkarr2d(int x, int y)
{
    struct arr2d a;
    a.arr = malloc(x * y * sizeof(float));
    a.dx = x;
    a.dy = y;
    return a;
}

