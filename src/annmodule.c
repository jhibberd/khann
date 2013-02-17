/* Python extension module for evaluating input vectors using an artificial
 * neural network implemented in C. */

#include <Python.h>
#include "config/digit2.h"

float *eval(float *iv);

static float *pylist_to_c_list(PyObject *args);
static PyObject *c_list_to_pylist(float *xs);

static PyObject *ann_eval(PyObject *self, PyObject *args) 
{
    float *i_vec = pylist_to_c_list(args);
    float *o_vec = eval(i_vec);
    free(i_vec);
    return c_list_to_pylist(o_vec);
}

static float *pylist_to_c_list(PyObject *args)
{
    PyObject *obj;
    PyObject *seq;
    int i, i_size;

    if (!PyArg_ParseTuple(args, "O", &obj))
        return NULL;
    seq = PySequence_Fast(obj, "Expected a sequence");
    i_size = PySequence_Size(obj);

    float *i_vec = malloc(i_size * sizeof(float));
    for (i = 0; i < i_size; i++) {
        *(i_vec+i) = (float) PyFloat_AsDouble(PySequence_Fast_GET_ITEM(seq, i));
    }

    Py_DECREF(seq);
    return i_vec;
}

static PyObject *c_list_to_pylist(float* xs) 
{
    PyObject *res;
    int i;

    res = PyList_New(NUM_OUTPUT_NODES);
    for (i = 0; i < NUM_OUTPUT_NODES; i++) 
        PyList_SetItem(res, i, Py_BuildValue("f", *xs++));

    return res;
}

static PyMethodDef ANNMethods[] = 
{
    {"eval", ann_eval, METH_VARARGS, "Evaluate an input vector."},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initann(void) 
{
    (void) Py_InitModule("ann", ANNMethods);
}

