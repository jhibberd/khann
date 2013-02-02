#include <Python.h>

static PyObject *spam_system(PyObject *self, PyObject *args)
{
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    return Py_BuildValue("i", sts);
}

// TODO(jhibberd) Experiment with static variables for lazy loading of network

static int flag = 0;

static PyObject *spam_eval(PyObject *self, PyObject *args)
{
    PyObject* obj;
    PyObject* seq;
    int i, len;

    if (!PyArg_ParseTuple(args, "O", &obj))
        return NULL;

    int sum = 0;
    long item;
    seq = PySequence_Fast(obj, "expected a sequence");
    len = PySequence_Size(obj);
    for (i = 0; i < len; i++) {
        item = PyInt_AsLong(PySequence_Fast_GET_ITEM(seq, i));
        sum += item;
    }
    Py_DECREF(seq);


    int x[3] = {7, 8, 9};
    PyObject* res;
    res = PyList_New(3);
    for (i = 0; i < 3; i++) 
        //PyList_SetItem(res, i, Py_BuildValue("i", x[i]));
        PyList_SetItem(res, i, Py_BuildValue("i", flag));

    flag = 1;

    //return Py_BuildValue("i", 3);
    return res;
}

static PyMethodDef SpamMethods[] = {
    {"system", spam_system, METH_VARARGS, "Execute a shell command."},
    {"eval", spam_eval, METH_VARARGS, "Evaluate an input vector.."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC
initspam(void)
{
    (void) Py_InitModule("spam", SpamMethods);
}
