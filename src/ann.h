
struct training_set {
    float **iv;     /* List of input vectors */
    float **ov;     /* List of output vectors */
    int n;          /* Number of elements in the training set */
    int size_iv;    /* Number of elements in each input vector */
    int size_ov;    /* Number of elements in each output vector */
};
