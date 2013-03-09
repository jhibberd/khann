
struct arr3d {
    double *arr;            /* Flattened array */
    int dx;                 /* Size of x dimension */
    int dy;                 /* Size of y dimension */
    int dz;                 /* Size of z dimension */
};

struct arr2d {
    double *arr;            /* Flattened array */
    int dx;                 /* Size of x dimension */
    int dy;                 /* Size of y dimension */
};

struct training_set {
    struct arr2d iv;        /* List of input vectors */
    struct arr2d ov;        /* List of output vectors */
    int n;                  /* Number of cases in the training set */
};

struct network {
    char id[20];            /* Network ID */
    struct arr2d output;    /* Output value of each node */
    struct arr2d error;     /* Error term value of each node */
    struct arr3d weight;    /* Weight value of each node */
    int layers;             /* Number of network layers */
    int topology[5];        /* Size of each network layer (max 5 layers) */
    double err_thresh;      /* The learning error threshold */
};

struct evaluation {
    double *ov;             /* Pointer to output vector */
    int n;                  /* Size of output vector */
};

/* Define how the weights in a network are initialised */
typedef enum {
    RAND_WEIGHTS,
    LOAD_WEIGHTS
} weight_mode;

/* Macros for setting and getting values in 2 and 3 dimensional arrays */
#define getarr3d(a, x, y, z) \
    &(a)->arr[((x) * (a)->dy * (a)->dz) + ((y) * (a)->dz) + (z)]
#define getarr2d(a, x, y) &(a)->arr[((x) * (a)->dy) + (y)]

#define sigmoid(x) 1 / (1 + expf(-x))
#define LEARNING_RATE 0.5
#define TEST_SAMPLE_SIZE 10

void cluster_init(void);
struct evaluation cluster_eval(const char *nid, double *iv);
void time_network(void);
void train_network(const char *nid);
void validate_network(const char *nid);

static struct arr2d mkarr2d(int x, int y);
static struct arr3d mkarr3d(int x, int y, int z);
static int did_classify(struct network *n, struct training_set *t, int ti);
static void free_network(struct network *n);
static void free_training_set(struct training_set *t);
static struct training_set load_training_set(struct network *n);
static void load_weights(struct network *n);
static void mknetwork(struct network *n, const char *nid, weight_mode wm); 
static void rand_weights(struct network *n);
static void save_weights(struct network *n);
static void set_error_terms(struct network *n, struct training_set *t, int ti);
static void set_outputs(struct network *n, double *iv);
static void set_weights(struct network *n);
static void test_weights(struct training_set *t, struct network *n);
/*static void time_train(struct training_set *t, struct network *n);*/
static void train(struct training_set *t, struct network *n);
static double training_error(struct network *n, struct training_set *t, int ti);

