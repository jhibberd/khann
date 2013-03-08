#include <Python.h>
#include "hashtable.h"

struct evaluation {
    double *ov;             /* Pointer to output vector */
    int n;                  /* Size of output vector */
};

extern void cluster_init(void);
extern struct evaluation cluster_eval(const char *nid, double *iv);

static PyObject *khann_cluster_init(PyObject *self, PyObject *noarg); 
static PyObject *khann_cluster_eval(PyObject *self, PyObject *args); 
static PyObject *khann_cluster_destroy(PyObject *self, PyObject *noarg); 

static PyMethodDef KhannMethods[] = {
    {
        "cluster_init", 
        khann_cluster_init, 
        METH_NOARGS, 
        "Initialise network cluster"
    },
    {
        "cluster_eval", 
        khann_cluster_eval, 
        METH_VARARGS, 
        "Evaluate an input vector using a network cluster"
    },
    {
        "cluster_destroy", 
        khann_cluster_destroy, 
        METH_NOARGS, 
        "Destroy a network cluster"
    },
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initkhann(void) 
{
    (void) Py_InitModule("khann", KhannMethods);
}

static PyObject *khann_cluster_init(PyObject *self, PyObject *noarg)
{
    cluster_init();
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *khann_cluster_eval(PyObject *self, PyObject *args) 
{
    int i;

    /* Convert C list to python list */
    PyObject *obj, *seq;
    int iv_size;
    double *iv;
    char *nid;

    if (!PyArg_ParseTuple(args, "sO", &nid, &obj))
        return NULL;

    seq = PySequence_Fast(obj, "Expected a sequence");
    iv_size = PySequence_Size(obj);
    iv = malloc(iv_size * sizeof(double));
    for (i = 0; i < iv_size; ++i)
        *(iv+i) = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(seq, i));
    Py_DECREF(seq);

    /* Evaluate input vector using C neural network */
    struct evaluation e;
    e = cluster_eval(nid, iv);
    free(iv);

    /* Convert evaluation result to python list */
    PyObject *res;
    res = PyList_New(e.n);
    for (i = 0; i < e.n; i++) 
        PyList_SetItem(res, i, Py_BuildValue("f", *e.ov++));

    return res;
}

static PyObject *khann_cluster_destroy(PyObject *self, PyObject *noarg)
{
    hashtable_destroy();
    Py_INCREF(Py_None);
    return Py_None;
}

