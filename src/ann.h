
struct arr3d {
    float *arr;             /* Flattened array */
    int dx;                 /* Size of x dimension */
    int dy;                 /* Size of y dimension */
    int dz;                 /* Size of z dimension */
};

struct arr2d {
    float *arr;             /* Flattened array */
    int dx;                 /* Size of x dimension */
    int dy;                 /* Size of y dimension */
};

struct training_set {
    float **iv;             /* List of input vectors */
    float **ov;             /* List of output vectors */
    int n;                  /* Number of elements in the training set */
    int size_iv;            /* Number of elements in each input vector */
    int size_ov;            /* Number of elements in each output vector */
};

struct network {
    struct arr2d output;    /* Output value of each node */
    struct arr2d error;     /* Error term value of each node */
    struct arr3d weight;    /* Weight value of each node */
    int layers;             /* Number of network layers */
};

/* Macros for setting and getting values in 2 and 3 dimensional arrays */
#define getarr3d(a, x, y, z) \
    &(a)->arr[((x) * (a)->dy * (a)->dz) + ((y) * (a)->dz) + (z)]
#define getarr2d(a, x, y) &(a)->arr[((x) * (a)->dy) + (y)]

struct arr3d mkarr3d(int x, int y, int z);
struct arr2d mkarr2d(int x, int y);
