/* Python extension module for evaluating input vectors using an artificial
 * neural network implemented in C. 
 */

#include <Python.h>
#include "config/digit.h"

struct eval_res {
    float *ov;              /* Pointer to output vector */
    int n;                  /* Size of output vector */
};

extern struct eval_res eval(float *iv);
static PyObject *ann_eval(PyObject *self, PyObject *args); 

/* Module manifest */
static PyMethodDef ANNMethods[] = 
{
    {"eval", ann_eval, METH_VARARGS, "Evaluate an input vector."},
    {NULL, NULL, 0, NULL}
};

/* Module initialisation */
PyMODINIT_FUNC initann(void) 
{
    (void) Py_InitModule("ann", ANNMethods);
}

/* Evaluate an input vector using C neural network */
static PyObject *ann_eval(PyObject *self, PyObject *args) 
{
    int i;

    /* Convert C list to python list */
    PyObject *obj;
    PyObject *seq;
    int iv_size;
    float *iv;
    double x;

    if (!PyArg_ParseTuple(args, "O", &obj))
        return NULL;
    seq = PySequence_Fast(obj, "Expected a sequence");
    iv_size = PySequence_Size(obj);
    iv = malloc(iv_size * sizeof(float));
    for (i = 0; i < iv_size; ++i) {
        x = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(seq, i));
        *(iv+i) = (float) x;
    }

    Py_DECREF(seq);

    /* Evaluate input vector using C neural network */
    struct eval_res o;
    o = eval(iv);
    free(iv);

    /* Convert evaluation result to python list */
    PyObject *res;
    res = PyList_New(o.n);
    for (i = 0; i < o.n; i++) 
        PyList_SetItem(res, i, Py_BuildValue("f", *o.ov++));

    return res;
}

