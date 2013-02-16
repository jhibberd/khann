#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "config/binadd.h"
#include "ann.h"

/* TODO(jhibberd) Get rid of this. Expanding it out is clearer */
#define final_output(n) getarr2d(&(n)->output, (n)->layers -1, 0)

static struct training_set load_training_set(void);
static void free_training_set(struct training_set *t);
static struct network mknetwork(void); 
static void free_network(struct network *n);
static float dotprod(float* a, float* b, int n);
static float sigmoid(float x);

const int topology[4] = TOPOLOGY;

/* void save_weights_to_file(void); */
void set_outputs(struct network *n, float *iv);
void set_weights(struct network *n);
float error(struct network *n, struct training_set *t, int i);
void set_error_terms(struct network *n, struct training_set *t, int i);

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
    /* TODO(jhibberd) Replace 'j' with 'i', the default iterator */
    int j, l;
    float o, *os, *ws;
   
    /* Set first layer of output nodes to value of input vector */ 
    for (j = 0; j < topology[0]; ++j)
        *getarr2d(&n->output, 0, j) = *iv++;

    /* Recursively set the output values of all downstream nodes, using the
     * current weight values */
    for (l = 1; l < n->layers; ++l)
        for (j = 0; j < topology[l]; ++j) {
            ws = getarr3d(&n->weight, l, j, 0);
            os = getarr2d(&n->output, l-1, 0);
            o = dotprod(ws, os, topology[l-1]);
            *getarr2d(&n->output, l, j) = sigmoid(o);
        } 
}

/* TODO(jhibberd) Add a doc string to each function */
/* TODO(jhibberd) Mark all functions as 'static' that are internal to this
 * translation unit */
void set_error_terms(struct network *n, struct training_set *t, int i) 
{
    int x;
    float *o, *tgt, *e;

    /* error term for output nodes */
    x = t->size_ov;
    e = getarr2d(&n->error, n->layers -1, 0);
    o = final_output(n);
    tgt = t->ov[i];
    while (x-- > 0) { 
        *e++ = *o * (1.0 - *o) * (*tgt++ - *o);
        o++;
        }

    int l, j;
    float er;
    /* set error terms for hidden nodes */
    for (l = n->layers -2; l >= 0; --l) {
        o = getarr2d(&n->output, l, 0);
        e = getarr2d(&n->error, l, 0);
        for (j = 0; j < topology[l]; j++) {

            /* TODO(jhibberd) This is complex and requires building another
             * array to try and preserve the use of dotprod. Rewrite using
             * the internals of dotprod and avoid need for new array. Also if
             * dotprod is only used in one other place just inline the code
             * and do away with the function */
            int k;
            float ws[topology[l+1]], w;
            for (k = 0; k < topology[l+1]; ++k) {
                w = *getarr3d(&n->weight, l+1, k, j);
                ws[k] = w;
            }
            
            float *err = getarr2d(&n->error, l+1, 0);
            er = dotprod(ws, err, topology[l+1]);
            *e++ = *o * (1.0 - *o) * er;
            o++;
        }
     }
}

/* TODO(jhibberd) Make LEARNING_RATE global define */
/* TODO(jhibberd) Remove project-specific defines no longer needed */
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

    o = final_output(n);
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

    o = final_output(n);
    tgt = t->ov[i];

    j = t->size_ov;
    while (j-- > 0)
        if (*tgt++ != roundf(*o++))
            return 0;
    
    return 1;
}

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

/* Compute the dot product of two n length arrays. */
static float dotprod(float* a, float* b, int n) 
{
    float s = 0;
    while (n-- > 0)
        s += *a++ * *b++;
    return s;
}

/* TODO(jhibberd) Macro */
static float sigmoid(float x) 
{
    return 1 / (1 + expf(-x));
}

