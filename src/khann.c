#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "hashtable.h"
#include "khann.h"
#include "mongo.h"


/* Learn weights that classify the training set with an acceptable level of
 * success */
void train_network(const char *nid)
{
    struct training_set t;
    struct network n;

    mknetwork(&n, nid, RAND_WEIGHTS);
    t = load_training_set(&n);
    train(&t, &n);
    test_weights(&t, &n);
    save_weights(&n);
    free_training_set(&t);
    free_network(&n);
}

/* Output the degree to which the saved weights correctly classify the training 
 * set */
void validate_network(const char *nid)
{
    struct training_set t;
    struct network n;

    mknetwork(&n, nid, LOAD_WEIGHTS);
    t = load_training_set(&n);
    test_weights(&t, &n);
    free_training_set(&t);
    free_network(&n);
}

/* Load all learnt networks into memory for on-demand evaluation of input
 * vectors */
void cluster_init(void)
{
    /* Get network IDs */
    char nids[10][256];
    int i;
    mongo conn[1];
    mongo_cursor cursor[1];
    int status;
    bson *b;
    bson_iterator it[1];

    status = mongo_client(conn, "127.0.0.1", 27017);
    if (status != MONGO_OK) {
        printf("Error connecting to database");
        exit(EXIT_FAILURE);
    }

    i = 0;
    mongo_cursor_init(cursor, conn, "khann__system.settings");
    while (mongo_cursor_next(cursor) == MONGO_OK) {
        b = &cursor->current;
        bson_iterator_init(it, b);
        bson_find(it, b, "_id");
        strcpy(nids[i++], bson_iterator_string(it));
    }

    mongo_cursor_destroy(cursor);
    mongo_destroy(conn);

    /* Initialise networks and store in hashtable */
    int j;
    struct network *n;
    for (j = 0; j < i; ++j) {
        n = (struct network *) malloc(sizeof(struct network));
        mknetwork(n, nids[j], LOAD_WEIGHTS);
        hashtable_set(nids[j], n);
    }
}

/* Evaluate an input vector using a network with pre-learnt weights. The return 
 * value is the network's output vector. This function is used by the python
 * extension module. */
struct evaluation cluster_eval(const char *nid, double *iv)
{
    struct network *n;
    struct evaluation e;

    n = (struct network *) hashtable_get(nid);
    set_outputs(n, iv);
    e.ov = getarr2d(&n->output, n->layers -1, 0);
    e.n = n->topology[n->layers -1];
    return e;
}

/* Construct a network, either with random weights or weights loaded from
 * a file */
static void mknetwork(struct network *n, const char *nid, weight_mode wm) 
{
    int i, max_layer_size;
    mongo conn[1];
    bson b[1], q[1];
    int status;

    /* Set network ID */
    strcpy(n->id, nid);

    /* Connect to database */
    status = mongo_client(conn, "127.0.0.1", 27017);
    if (status != MONGO_OK) {
        printf("Error connecting to database");
        exit(EXIT_FAILURE);
    }

    /* Read settings for network */
    bson_init(q);
    bson_append_string(q, "_id", nid);
    bson_finish(q);
    status = mongo_find_one(conn, "khann__system.settings", q, NULL, b);
    if (status != MONGO_OK) {
        fprintf(stderr, "Failed to find '%s' settings doc in database", nid);
        exit(EXIT_FAILURE);
    }
   
    bson_iterator it[1], sub[1];
    bson_iterator_init(it, b);

    /* Set topology and number of layers */
    bson_iterator_init(it, b);
    if (!bson_find(it, b, "topology")) {
        printf("Corrupt settings doc");
        exit(EXIT_FAILURE);
    }
    bson_iterator_subiterator(it, sub);
    n->layers = 0;
    while (bson_iterator_next(sub) != BSON_EOO) {
        n->topology[n->layers] = bson_iterator_int(sub);
        ++n->layers;
    }

    /* Set error threshold */
    if (!bson_find(it, b, "error_threshold")) {
        printf("Corrupt settings doc");
        exit(EXIT_FAILURE);
    }
    n->err_thresh = bson_iterator_double(it); 

    bson_destroy(q);
    bson_destroy(b);
    mongo_destroy(conn);

    /* Get size of largest layer (all arrays will be 'square' for now) */
    max_layer_size = 0;
    for (i = 0; i < n->layers; i++)
        if (n->topology[i] > max_layer_size)
            max_layer_size = n->topology[i];

    /* Allocate memory for all arrays */
    n->output =  mkarr2d(n->layers, max_layer_size); 
    n->error =   mkarr2d(n->layers, max_layer_size); 
    n->weight =  mkarr3d(n->layers, max_layer_size, max_layer_size);

    /* Set weights, either by randomly assigning values if the network is 
     * being trained, or by loading values if the network has already been
     * trained */
    switch (wm) {
        case RAND_WEIGHTS:
            rand_weights(n);
            break;
        case LOAD_WEIGHTS:
            load_weights(n);
            break;
    }
}

/* Assign random weights (between -0.5 and +0.5) to the network */
static void rand_weights(struct network *n)
{
    int i, j, k;
    double w;

    srand(time(NULL));
    for (i = 1; i < n->layers; ++i)
        for (j = 0; j < n->topology[i]; ++j)
            for (k = 0; k < n->topology[i-1]; ++k) {
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
    for (i = 0; i < n->topology[0]; ++i)
        *getarr2d(&n->output, 0, i) = *iv++;

    /* Recursively set the output values of all downstream nodes, using the
     * current weight values */
    for (i = 1; i < n->layers; ++i)
        for (j = 0; j < n->topology[i]; ++j) {

            double dp = 0, *pw, *po;
            int c;

            pw = getarr3d(&n->weight, i, j, 0);
            po = getarr2d(&n->output, i-1, 0);
            c = n->topology[i-1];
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
        for (j = 0; j < n->topology[i]; ++j) {

            int k;
            double dp = 0, w, *e2;
            
            e2 = getarr2d(&n->error, i+1, 0);
            for (k = 0; k < n->topology[i+1]; ++k) {
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
        for (j = 0; j < n->topology[i]; ++j) {
            w = getarr3d(&n->weight, i, j, 0);
            o = getarr2d(&n->output, i-1, 0); 
            f = LEARNING_RATE * *getarr2d(&n->error, i, j);
            c = n->topology[i-1];
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
    int i, cfd;
    double err, *iv;

    fprintf(stderr, "Training with training set of %d items\n", t->n);
    if (t->n == 0)
        return;

    do {
        err = 0, cfd = 0;

        for (i = 0; i < t->n; ++i) {
            iv = getarr2d(&t->iv, i, 0);
            set_outputs(n, iv);
            err += training_error(n, t, i);
            cfd += did_classify(n, t, i);
            set_error_terms(n, t, i);
            set_weights(n);
        }

        /* Progress output */
        fprintf(
            stderr, "error %f/%f | classified %d/%d\n", 
            (err / t->n), n->err_thresh, cfd, t->n);

    } while ((err / t->n) > n->err_thresh);
}

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

    if (t->n == 0)
        return;

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
static struct training_set load_training_set(struct network *n)
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
    int size_iv, size_ov;
    char db[256];
    sprintf(db, "khann_%s", n->id);
    t.n = (int) mongo_count(conn, db, "training", NULL);
    size_iv = n->topology[0];
    size_ov = n->topology[n->layers-1];

    /* Allocate enough memory for an array to hold all input and output vectors
     * of the training set */
    t.iv = mkarr2d(t.n, size_iv); 
    t.ov = mkarr2d(t.n, size_ov); 

    /* Iterate over all training cases in the database and load each one into
     * memory */
    double *iv, *ov;
    int i;
    char ns[256];
    mongo_cursor cursor[1];
    sprintf(ns, "khann_%s.training", n->id);
    mongo_cursor_init(cursor, conn, ns);
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

/* Save network weights to a file */
static void save_weights(struct network *n) 
{
    int num;
    FILE *fp;
    char path[1024];

    sprintf(path, "/home/jhibberd/projects/khann/weights/%s", n->id);
    fp = fopen(path, "w");
    num = n->weight.dx * n->weight.dy * n->weight.dz;
    fwrite(n->weight.arr, sizeof(double), num, fp); 
    fclose(fp);
}

/* Load network weights from a file */
static void load_weights(struct network *n) 
{
    int num;
    FILE *fp;
    char path[1024];

    sprintf(path, "/home/jhibberd/projects/khann/weights/%s", n->id);
    fp = fopen(path, "r");
    num = n->weight.dx * n->weight.dy * n->weight.dz;
    if (fread(n->weight.arr, sizeof(double), num, fp) != num) {
        fprintf(stderr, "Failed to read weights from file");
        exit(EXIT_FAILURE);
    }
    fclose(fp);
}

