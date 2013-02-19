#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "config/binadd.h"
#include "ann.h"

static const int topology[] = TOPOLOGY;

/* Learn weights that classify the training set with an acceptable level of
 * success */
void train_network(void)
{
    struct training_set t;
    struct network n;

    t = load_training_set();
    n = mknetwork(RAND_WEIGHTS);
    train(&t, &n);
    test_weights(&t, &n);
    save_weights(&n);
    free_training_set(&t);
    free_network(&n);
}

/* Output the degree to which the saved weights correctly classify the training 
 * set */
void validate_network(void)
{
    struct training_set t;
    struct network n;

    t = load_training_set();
    n = mknetwork(LOAD_WEIGHTS);
    test_weights(&t, &n);
    free_training_set(&t);
    free_network(&n);
}

/* Evaluate an input vector using a network with pre-learnt weights. The return 
 * value is the network's output vector. This function is used by the python
 * extension module. */
float *eval(float *iv) 
{
    /* Lazily initialise the network, but once loaded persist across calls in 
     * static storage */
    static short loaded = 0;
    static struct network n;
    if (!loaded) {
        n = mknetwork(LOAD_WEIGHTS);
        loaded = 1;
    }

    set_outputs(&n, iv);
    return getarr2d(&n.output, n.layers -1, 0);
}

static struct network mknetwork(weight_mode wm) 
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

    /* Set weights, either by randomly assigning values if the network is 
     * being trained, or by loading values if the network has already been
     * trained */
    switch (wm) {
        case RAND_WEIGHTS:
            rand_weights(&n);
            break;
        case LOAD_WEIGHTS:
            load_weights(&n);
            break;
    }

    return n;
}

/* Assign random weights (between -0.5 and +0.5) to the network */
static void rand_weights(struct network *n)
{
    int i, j, k;
    float w;

    /* TODO(jhibberd) Revert */
    /*srand(time(NULL));*/
    srand(13);
    for (i = 1; i < n->layers; ++i)
        for (j = 0; j < topology[i]; ++j)
            for (k = 0; k < topology[i-1]; ++k) {
                w = ((double)rand() / (double)RAND_MAX) - 0.5;
                *getarr3d(&n->weight, i, j, k) = w;
            }
}

static void set_outputs(struct network *n, float *iv) 
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
static void set_error_terms(struct network *n, struct training_set *t, int ti) 
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

static void set_weights(struct network *n) 
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

/* Return the difference between the target output vector (z) and actual output
 * vector (o) as an error score (e) */
static float training_error(struct network *n, struct training_set *t, int ti) 
{
    int c;
    float e, *o, *z;

    o = getarr2d(&n->output, n->layers -1, 0);
    z = t->ov[ti];
    c = t->size_ov;
    e = 0;
    while (c-- > 0)
        e += pow((*z++ - *o++), 2);

    return e * 0.5;
}

/* Repeatedly apply the entire training set to the network, adjusting the
 * weights after each member of the training set, until the error over the
 * entire training set falls below a defined threshold */
static void train(struct training_set *t, struct network *n)
{
    int i, it, cfd;
    float err;

    fprintf(stderr, "Training with training set of %d items\n", t->n);

    it = 0;
    do {
        err = 0, cfd = 0;

        for (i = 0; i < t->n; ++i) {
            set_outputs(n, t->iv[i]);
            err += training_error(n, t, i);
            if (it == DEBUG_THRESHOLD)
                cfd += did_classify(n, t, i);
            set_error_terms(n, t, i);
            set_weights(n);
        }

        /* Progress output */
        if (it == DEBUG_THRESHOLD) {
            fprintf(
                stderr, "error %f/%f | classified %d/%d\n", 
                (err / t->n), ERROR_THRESHOLD, cfd, t->n);
            it = 0;
        }
        else 
            ++it;

    } while ((err / t->n) > ERROR_THRESHOLD);
}

/* Return whether the network produced an output vector (o), that when rounded,
 * was identical to the training set (z), whose values are always binary, not
 * float */
static int did_classify(struct network *n, struct training_set *t, int ti) 
{
    int c;
    float *o, *z;

    o = getarr2d(&n->output, n->layers -1, 0);
    z = t->ov[ti];
    c = t->size_ov;
    while (c-- > 0)
        if (*z++ != roundf(*o++))
            return 0;
    
    return 1;
}

/* Test the weights by applying them to the training set and print the number
 * of correct classifications. A random sample of tests is also printed. */
static void test_weights(struct training_set *t, struct network *n) 
{
    int i, c, ti, cfd;
    float o;

    /* Print number of correct classifications */
    cfd = 0;
    for (i = 0; i < t->n; ++i) {
        set_outputs(n, t->iv[i]);
        cfd += did_classify(n, t, i);
    }
    fprintf(stderr, "classified %d/%d\n", cfd, t->n);

    /* Print sample training set tests */
    srand(time(NULL));
    c = TEST_SAMPLE_SIZE; 
    while (c-- > 0) {

        ti = rand() % t->n; 

        for (i = 0; i < t->size_iv; i++)
            fprintf(stderr, "%.0f ", t->iv[ti][i]);
        fprintf(stderr, "-> ");
        for (i = 0; i < t->size_ov; i++)
            fprintf(stderr, "%.0f ", t->ov[ti][i]);
        fprintf(stderr, "| ");

        set_outputs(n, t->iv[ti]);
        for (i = 0; i < t->size_ov; i++) {
            o = *getarr2d(&n->output, n->layers -1, i);
            fprintf(stderr, "%f ", o);
        }
        fprintf(stderr, "\n");

    }
}

static struct training_set load_training_set(void)
{
    struct training_set t;

    /* Make an unbuffered pass through the training set file to count the 
     * number of elements */
    t.n = 0;
    int ch;
    FILE *fp = fopen("data/" DATA_KEY ".training", "r");
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
static struct arr3d mkarr3d(int x, int y, int z)
{
    struct arr3d a;
    a.arr = malloc(x * y * z * sizeof(float));
    a.dx = x;
    a.dy = y;
    a.dz = z;
    return a;
}

/* Construct (and allocate memory for) a 2D array */
static struct arr2d mkarr2d(int x, int y)
{
    struct arr2d a;
    a.arr = malloc(x * y * sizeof(float));
    a.dx = x;
    a.dy = y;
    return a;
}

/* Save network weights to a file */
static void save_weights(struct network *n) 
{
    int i, j, k;
    float w;
    FILE *fp;

    fp = fopen("data/" DATA_KEY ".weights", "w");

    for (i = 1; i < n->layers; ++i)
        for (j = 0; j < topology[i]; ++j)
            for (k = 0; k < topology[i-1]; ++k) {
                w = *getarr3d(&n->weight, i, j, k);
                fprintf(fp, "%f,", w);
            }

    fclose(fp);
}

/* Load network weights from a file */
static void load_weights(struct network *n) 
{
    int i, j, k;
    float w;
    FILE *fp;

    fp = fopen("data/" DATA_KEY ".weights", "r");

    for (i = 1; i < n->layers; ++i)
        for (j = 0; j < topology[i]; ++j)
            for (k = 0; k < topology[i-1]; ++k) {
                if (fscanf(fp, "%f,", &w) != 1) {
                    fprintf(stderr, "%s", "Error loading weights");
                    exit(1);
                }
                *getarr3d(&n->weight, i, j, k) = w;
            }

    fclose(fp);
}

