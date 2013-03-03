#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "config/digit.h"
#include "ann.h"
#include "mongo.h"

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

/* Time how long it takes to test one training case in the network /
void time_network(void)
{
    struct training_set t;
    struct network n;

    t = load_training_set();
    n = mknetwork(RAND_WEIGHTS);
    time_train(&t, &n);
    free_training_set(&t);
    free_network(&n);
}*/

/* Evaluate an input vector using a network with pre-learnt weights. The return 
 * value is the network's output vector. This function is used by the python
 * extension module. */
struct eval_res eval(double *iv) 
{
    /* Lazily initialise the network, but once loaded persist across calls in 
     * static storage */
    static short loaded = 0;
    static struct network n;
    if (!loaded) {
        n = mknetwork(LOAD_WEIGHTS);
        loaded = 1;
    }

    struct eval_res res;
    set_outputs(&n, iv);
    res.ov = getarr2d(&n.output, n.layers -1, 0);
    res.n = topology[n.layers -1];
    return res;
}

/* Construct a network, either with random weights or weights loaded from
 * a file */
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
    double w;

    srand(time(NULL));
    for (i = 1; i < n->layers; ++i)
        for (j = 0; j < topology[i]; ++j)
            for (k = 0; k < topology[i-1]; ++k) {
                w = ((double)rand() / (double)RAND_MAX) - 0.5;
                *getarr3d(&n->weight, i, j, k) = w;
            }
}

/* Load an input vector into the network then recursively set all output node
 * values */
static void set_outputs(struct network *n, double *iv) 
{
    int i, j;

    /* Set first layer of output nodes to value of input vector */ 
    for (i = 0; i < topology[0]; ++i)
        *getarr2d(&n->output, 0, i) = *iv++;

    /* Recursively set the output values of all downstream nodes, using the
     * current weight values */
    for (i = 1; i < n->layers; ++i)
        for (j = 0; j < topology[i]; ++j) {

            double dp = 0, *pw, *po;
            int c;

            pw = getarr3d(&n->weight, i, j, 0);
            po = getarr2d(&n->output, i-1, 0);
            c = topology[i-1];
            while (c-- > 0)
                dp += *pw++ * *po++;

            *getarr2d(&n->output, i, j) = sigmoid(dp);
        } 
}

/* Working backwards through the network set the error term value for each
 * node based on the difference between the actual network output and the
 * expected network output (from the training set) */
static void set_error_terms(struct network *n, struct training_set *t, int ti) 
{
    int i, j;
    double *o, *e;

    /* Set the error terms (e) for the final output nodes (o), based on the 
     * expected values (z) in the training set (t) */
    double *z;
    i = t->ov.dy;
    e = getarr2d(&n->error, n->layers -1, 0);
    o = getarr2d(&n->output, n->layers -1, 0);
    z = getarr2d(&t->ov, ti, 0);
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
            double dp = 0, w, *e2;
            
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

/* Adjust the weight values of all network nodes based on error term and output
 * values */
static void set_weights(struct network *n) 
{
    int i, j, c;
    double *w, *o, f;

    for (i = 1; i < n->layers; ++i)
        for (j = 0; j < topology[i]; ++j) {
            w = getarr3d(&n->weight, i, j, 0);
            o = getarr2d(&n->output, i-1, 0); 
            f = LEARNING_RATE * *getarr2d(&n->error, i, j);
            c = topology[i-1];
            while (c-- > 0)
                *w++ += f * *o++;
        }
}

/* Return the difference between the target output vector (z) and actual output
 * vector (o) as an error score (e) */
static double training_error(struct network *n, struct training_set *t, int ti) 
{
    int c;
    double e, *o, *z;

    o = getarr2d(&n->output, n->layers -1, 0);
    z = getarr2d(&t->ov, ti, 0);
    c = t->ov.dy;
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
    double err, *iv;

    fprintf(stderr, "Training with training set of %d items\n", t->n);

    it = 0;
    do {
        err = 0, cfd = 0;

        for (i = 0; i < t->n; ++i) {
            iv = getarr2d(&t->iv, i, 0);
            set_outputs(n, iv);
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

/* TODO(jhibberd) With "--std-c99" various time components no longer exist.
 * Needs to be rewritten using the standard */
/* Time how long it takes on average to train the network using a single
 * training case /
static void time_train(struct training_set *t, struct network *n)
{
    const int test_size = 1000;
    int i, dt;
    float per_tc;
    struct timespec tm_before, tm_after;

    fprintf(stderr, "Testing network training time:\n");

    clock_gettime(CLOCK_MONOTONIC, &tm_before); 
    for (i = 0; i < test_size; ++i) {
        set_outputs(n, t->iv[i]);
        training_error(n, t, i);
        set_error_terms(n, t, i);
        set_weights(n);
    }
    clock_gettime(CLOCK_MONOTONIC, &tm_after);

    dt = (int) (tm_after.tv_sec - tm_before.tv_sec);
    per_tc = ((float) dt) / ((float) test_size);
    fprintf(stderr, "\t%f seconds(s) per training case\n", per_tc);
    fprintf(stderr, "\t%f seconds(s) for training set (%i cases)\n", 
            per_tc * t->n, t->n);
}
*/

/* Return whether the network produced an output vector (o), that when rounded,
 * was identical to the training set (z), whose values are always binary, not
 * float */
static int did_classify(struct network *n, struct training_set *t, int ti) 
{
    int c;
    double *o, *z;

    o = getarr2d(&n->output, n->layers -1, 0);
    z = getarr2d(&t->ov, ti, 0);
    c = t->ov.dy;
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
    double o, *iv, x;

    /* Print number of correct classifications */
    cfd = 0;
    for (i = 0; i < t->n; ++i) {
        iv = getarr2d(&t->iv, i, 0);
        set_outputs(n, iv);
        cfd += did_classify(n, t, i);
    }
    fprintf(stderr, "classified %d/%d\n", cfd, t->n);

    /* Print sample training set tests */
    srand(time(NULL));
    c = TEST_SAMPLE_SIZE; 
    while (c-- > 0) {

        ti = rand() % t->n; 

        for (i = 0; i < t->iv.dy; i++) {
            x = *getarr2d(&t->iv, ti, i);
            fprintf(stderr, "%.0f ", x);
            }
        fprintf(stderr, "-> ");
        for (i = 0; i < t->ov.dy; i++) {
            x = *getarr2d(&t->ov, ti, i);
            fprintf(stderr, "%.0f ", x);
            }
        fprintf(stderr, "| ");

        iv = getarr2d(&t->iv, ti, 0);
        set_outputs(n, iv);
        for (i = 0; i < t->ov.dy; i++) {
            o = *getarr2d(&n->output, n->layers -1, i);
            fprintf(stderr, "%f ", o);
        }
        fprintf(stderr, "\n");

    }
}

/* Load the training set from file into memory */
static struct training_set load_training_set(void)
{
    struct training_set t;

    /* Establish connection */
    mongo conn[1];
    int status = mongo_client(conn, "127.0.0.1", 27017);
    if (status != MONGO_OK) {
        printf("Error connecting to database");
        exit(EXIT_FAILURE);
    }

    /* Discover the dimensions of the training set */
    int size_iv, size_ov, layers;
    t.n = (int) mongo_count(conn, "khann_" DATA_KEY, "training", NULL);
    layers = sizeof(topology) / sizeof(int);
    size_iv = topology[0];
    size_ov = topology[layers-1];

    /* Allocate enough memory for an array to hold all input and output vectors
     * of the training set */
    t.iv = mkarr2d(t.n, size_iv); 
    t.ov = mkarr2d(t.n, size_ov); 

    /* Iterate over all training cases in the database and load each one into
     * memory */
    double *iv, *ov;
    int i;
    mongo_cursor cursor[1];
    mongo_cursor_init(cursor, conn, "khann_" DATA_KEY ".training");
    i = 0;
    iv = t.iv.arr;
    ov = t.ov.arr;

    while (i < t.n) {
        if (mongo_cursor_next(cursor) != MONGO_OK) {
            printf("Failed to iterate through training set in database");
            exit(EXIT_FAILURE);
        }

        bson_iterator it[1], sub[1];
        bson *b = &cursor->current;
        bson_iterator_init(it, b);

        bson_find(it, b, "iv");
        bson_iterator_subiterator(it, sub);
        while (bson_iterator_next(sub) != BSON_EOO)
            *iv++ = bson_iterator_double(sub);

        bson_find(it, b, "ov");
        bson_iterator_subiterator(it, sub);
        while (bson_iterator_next(sub) != BSON_EOO)
            *ov++ = bson_iterator_double(sub);

        ++i;
    }

    mongo_cursor_destroy(cursor);
    mongo_destroy(conn);
    return t;
}

/* Free memory dynamically allocated for the training set */
static void free_training_set(struct training_set *t)
{
    free(t->iv.arr);
    free(t->ov.arr);
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
    a.arr = malloc(x * y * z * sizeof(double));
    a.dx = x;
    a.dy = y;
    a.dz = z;
    return a;
}

/* Construct (and allocate memory for) a 2D array */
static struct arr2d mkarr2d(int x, int y)
{
    struct arr2d a;
    a.arr = malloc(x * y * sizeof(double));
    a.dx = x;
    a.dy = y;
    return a;
}

/* Save network weights to the database */
static void save_weights(struct network *n) 
{
    mongo conn[1];
    bson b[1];
    int status, key_i;
    char key[10];
    int i, j, k;
    double w;

    status = mongo_client(conn, "127.0.0.1", 27017);
    if (status != MONGO_OK) {
        printf("Error connecting to database");
        exit(EXIT_FAILURE);
    }
    
    bson_init(b);
    bson_append_string(b, "_id", DATA_KEY);
    bson_append_start_array(b, "data");

    key_i = 0;
    for (i = 1; i < n->layers; ++i)
        for (j = 0; j < topology[i]; ++j)
            for (k = 0; k < topology[i-1]; ++k) {
                w = *getarr3d(&n->weight, i, j, k);
                sprintf(key, "%d", key_i);
                bson_append_double(b, key, w);
                ++key_i;
            }

    bson_append_finish_array(b);
    bson_finish(b);
    mongo_insert(conn, "khann__system.weights", b, NULL);
    bson_destroy(b);
    mongo_destroy(conn);
}

/* Load network weights from the database */
static void load_weights(struct network *n) 
{
    mongo conn[1];
    bson b[1], q[1];
    int i, j, k;
    double w;
    int status;

    status = mongo_client(conn, "127.0.0.1", 27017);
    if (status != MONGO_OK) {
        printf("Error connecting to database");
        exit(EXIT_FAILURE);
    }

    bson_init(q);
    bson_append_string(b, "_id", DATA_KEY);
    bson_finish(q);
    status = mongo_find_one(conn, "khann__system.weights", q, NULL, b);
    if (status != MONGO_OK) {
        printf("Failed to find weights doc in database");
        exit(EXIT_FAILURE);
    }
    
    bson_iterator it[1], sub[1];
    bson_iterator_init(it, b);
    if (!bson_find(it, b, "data")) {
        printf("Corrupt weights doc");
        exit(EXIT_FAILURE);
    }
    bson_iterator_subiterator(it, sub);

    for (i = 1; i < n->layers; ++i)
        for (j = 0; j < topology[i]; ++j)
            for (k = 0; k < topology[i-1]; ++k) {
                bson_iterator_next(sub);
                w = bson_iterator_double(sub);
                printf("%fl-", w);
                *getarr3d(&n->weight, i, j, k) = w;
            }

    bson_destroy(q);
    bson_destroy(b);
}

