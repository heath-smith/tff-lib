/* Copyright 2023 Heath Smith
 *
 * This module contains the PyObject
 * OpticalMedium, which is a C++ implementation
 * of the same class from tff_lib. This module
 * relies on the CPython API. (Python.h)
 *
 *
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#include <math.h>
#include <stdexcept>
#include <iostream>
#include <complex>

// this is here to track changes to the API
static const char* PICKLE_VERSION_KEY = "_pickle_version";
static int PICKLE_VERSION = 1;

// declare the object extension type
typedef struct {
  PyObject_HEAD
  PyObject *waves;   /* NDArray, wavelengths */
  PyObject *nref;    /* NDArray, refrective indices */
  PyObject *_thick;  /* float, thickness */
  PyObject *_ntype;  /* int, index type -- high, low, unspecified */
} OpticalMedium;

/* First, the traversal method lets the cyclic GC know about subobjects
 * that could participate in cycles. For each subobject that can participate
 * in cycles, we need to call the visit() function, which is passed to the
 * traversal method. The visit() function takes as arguments the subobject
 * and the extra argument arg passed to the traversal method. It returns an
 * integer value that must be returned if it is non-zero.
 *
 * Python provides a Py_VISIT() macro that automates calling visit functions.
 * With Py_VISIT(), we can minimize the amount of boilerplate in Custom_traverse
 */
static int
OpticalMedium_traverse(OpticalMedium *self, visitproc visit, void *arg)
{
    Py_VISIT(self->waves);
    Py_VISIT(self->nref);
    Py_VISIT(self->_thick);
    Py_VISIT(self->_ntype);
    return 0;
}

/*
 * Second, we need to provide a method for clearing any subobjects that
 * can participate in cycles:
 *
 * Notice the use of the Py_CLEAR() macro. It is the recommended and safe
 * way to clear data attributes of arbitrary types while decrementing their
 * reference counts. If you were to call Py_XDECREF() instead on the
 * attribute before setting it to NULL, there is a possibility that the
 * attribute’s destructor would call back into code that reads the attribute
 * again (especially if there is a reference cycle).
 */
static int
OpticalMedium_clear(OpticalMedium *self)
{
    Py_CLEAR(self->waves);
    Py_CLEAR(self->nref);
    Py_CLEAR(self->_thick);
    Py_CLEAR(self->_ntype);
    return 0;
}


/* Because we have data to manage, we must consider
 * object allocation and deallocation, so we supply
 * a deallocation method.
 *
 * The deallocator Custom_dealloc may call arbitrary
 * code when clearing attributes. It means the circular
 * GC can be triggered inside the function. Since the
 * GC assumes reference count is not zero, we need to
 * untrack the object from the GC by calling PyObject_GC_UnTrack()
 * before clearing members. Here is our reimplemented deallocator
 * using PyObject_GC_UnTrack() and Custom_clear:
 */
static void OpticalMedium_dealloc(OpticalMedium *self) {
  PyObject_GC_UnTrack(self);
  OpticalMedium_clear(self);
  Py_TYPE(self)->tp_free((PyObject *) self);
}

/* We provide a tp_new implementation to ensure
 * that the class members are initialized properly.
 */
static PyObject *
OpticalMedium_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
  OpticalMedium *self;
  // call tp_alloc slot to allocate memory
  self = (OpticalMedium *) type->tp_alloc(type, 0);
  if (self != NULL) {
    self->waves = NULL;
    self->nref = NULL;
    self->_thick = PyLong_FromLong(-1);  // initialize to -1
    if (self->_thick == NULL) {
      Py_DECREF(self);
    }
    self->_ntype = PyLong_FromLong(-1);    // initialize to -1
    if (self->_ntype == NULL) {
      Py_DECREF(self);
    }
  }
  return (PyObject *) self;
}

/* Define the initialization function, __init__()
 */
static int
OpticalMedium_init(OpticalMedium *self, PyObject *args, PyObject *kwds) {
  static char *arglist[] = {
    "waves", "nref", "thick", "ntype", NULL };
  PyObject *waves = NULL, *nref = NULL, *thick = NULL, *ntype = NULL;
  PyObject *tmp = NULL, *temp_wv = NULL, *temp_nref = NULL;
  int DTYPE;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|OO", arglist,
                                    &waves, &nref, &thick, &ntype))
    return -1;

  if (waves) {

    // wavelength values should always be floating point numbers
    temp_wv = PyArray_FROM_OTF(waves, NPY_FLOAT64, NPY_ARRAY_INOUT_ARRAY);

    if (temp_wv == NULL) {
      Py_CLEAR(temp_wv);
      return -1;
    }

    if (PyArray_NDIM((PyArrayObject *)temp_wv) != 1) {
      Py_CLEAR(temp_wv);
      PyErr_SetString(PyExc_ValueError, "waves expects 1-dimensional array");
      return -1;
    }

  }

  if (nref) {

    // use complex values for NREF, ok if doubles or ints passed in
    temp_nref = PyArray_FROM_OTF(nref, NPY_COMPLEX128, NPY_ARRAY_INOUT_ARRAY);

    if (temp_nref == NULL) {
      Py_CLEAR(temp_nref);
      return -1;
    }

    if (PyArray_NDIM((PyArrayObject *)temp_nref) != 1) {
      Py_CLEAR(temp_nref);
      PyErr_SetString(PyExc_ValueError, "nref expects 1-dimensional array");
      return -1;
    }

  }

  // verify both array's have same number of elements
  if (PyArray_SIZE((PyArrayObject *)temp_wv) != PyArray_SIZE((PyArrayObject *)temp_nref)) {
    Py_CLEAR(temp_wv);
    Py_CLEAR(temp_nref);
    PyErr_SetString(
      PyExc_ValueError, "waves and nref must have the same number of elements");
    return -1;
  }

  self->waves = Py_BuildValue("N", temp_wv);
  self->nref = Py_BuildValue("N", temp_nref);

  Py_INCREF(self->waves);
  Py_INCREF(self->nref);

  Py_XDECREF(temp_wv);
  Py_XDECREF(temp_nref);

  if (thick) {

    if (PyFloat_AsDouble(thick) < 0) {
      if (PyFloat_AsDouble(thick) != -1) {
        PyErr_SetString(PyExc_ValueError, "thick must be > 0 or -1 for infinite medium");
        return -1;
      }
    }

    tmp = self->_thick;
    Py_INCREF(thick);
    self->_thick = thick;
    Py_XDECREF(tmp);
  }

  if (ntype) {

    tmp = self->_ntype;
    Py_INCREF(ntype);
    self->_ntype = ntype;
    Py_XDECREF(tmp);
  }

  return 0;
}

/* __getstate__ pickles the object. __getstate__ is expected to
 * return a dictionary of the internal state of the Custom object.
 * Note that a Custom object has two Python objects (first and last)
 * and a C integer (number) that need to be converted to a Python object.
 */
static PyObject *
OpticalMedium_getstate(OpticalMedium *self, PyObject *Py_UNUSED(ignored)) {
  PyObject *ret = Py_BuildValue("{sOsOsOsOsi}",
                                "waves", self->waves,
                                "nref", self->nref,
                                "_thick", self->_thick,
                                "_ntype", self->_ntype,
                                PICKLE_VERSION_KEY, PICKLE_VERSION);
  return ret;
}

/* The implementation of __setstate__ un-pickles the object. This is a
 * little more complicated as there is quite a lot of error checking going
 * on. We are being passed an arbitrary Python object and need to check:
 *
 *   -It is a Python dictionary.
 *
 *   -It has a version key and the version value is one that we can deal with.
 *
 *   -It has the required keys and values to populate our Custom object.
 *
 * Note that our __new__ method (Custom_new()) has already been called on self.
 * Before setting any member value we need to de-allocate the existing value
 * set by Custom_new() otherwise we will have a memory leak.
 */
static PyObject *
OpticalMedium_setstate(OpticalMedium *self, PyObject *state) {
  /* Error checking */
  if (!PyDict_CheckExact(state)) {
    PyErr_SetString(PyExc_ValueError, "Pickled object is not a dict.");
    return NULL;
  }

  /* Check version
   * Borrowed reference but no need to increment as we
   * create a C long * from it.
   */
  PyObject *temp = PyDict_GetItemString(state, PICKLE_VERSION_KEY);
  if (temp == NULL) {
    /* PyDict_GetItemString does not set any error state so we have to. */
    PyErr_Format(PyExc_KeyError,  "No \"%s\" in pickled dict.", PICKLE_VERSION_KEY);
    return NULL;
  }

  int pickle_version = (int) PyLong_AsLong(temp);
  if (pickle_version != PICKLE_VERSION) {
    PyErr_Format(PyExc_ValueError,
                 "Pickle version mismatch. Got version %d but expected version %d.",
                 pickle_version, PICKLE_VERSION);
    return NULL;
  }

  /* Set waves member */
  /* NOTE: Custom_new() will have been invoked so self->first and self->last
  * will have been allocated so we have to de-allocate them. */
  Py_XDECREF(self->waves);
  self->waves = PyDict_GetItemString(state, "waves"); /* Borrowed reference. */
  if (self->waves == NULL) {
      /* PyDict_GetItemString does not set any error state so we have to. */
      PyErr_SetString(PyExc_KeyError, "No \"waves\" in pickled dict.");
      return NULL;
  }
  /* Increment the borrowed reference for our instance of it. */
  Py_INCREF(self->waves);


  /* Set nref member */
  Py_XDECREF(self->nref);
  self->nref = PyDict_GetItemString(state, "nref"); /* Borrowed reference. */
  if (self->nref == NULL) {
      /* PyDict_GetItemString does not set any error state so we have to. */
      PyErr_SetString(PyExc_KeyError, "No \"nref\" in pickled dict.");
      return NULL;
  }
  Py_INCREF(self->nref);

  /* Set _thick member */
  Py_XDECREF(self->_thick);
  self->_thick = PyDict_GetItemString(state, "_thick"); /* Borrowed reference. */
  if (self->_thick == NULL) {
      /* PyDict_GetItemString does not set any error state so we have to. */
      PyErr_SetString(PyExc_KeyError, "No \"_thick\" in pickled dict.");
      return NULL;
  }
  Py_INCREF(self->_thick);

  /* Set _ntype member */
  Py_XDECREF(self->_ntype);
  self->_ntype = PyDict_GetItemString(state, "_ntype"); /* Borrowed reference. */
  if (self->_ntype == NULL) {
      /* PyDict_GetItemString does not set any error state so we have to. */
      PyErr_SetString(PyExc_KeyError, "No \"_ntype\" in pickled dict.");
      return NULL;
  }
  Py_INCREF(self->_ntype);

  Py_RETURN_NONE;

}

static PyMemberDef OpticalMedium_members[] = {
  {"waves", T_OBJECT_EX, offsetof(OpticalMedium, waves), 0, "waves"},
  {"nref", T_OBJECT_EX, offsetof(OpticalMedium, nref), 0, "nref"},
  {NULL}  /* Sentinel */
};

/* Provide customized getters/setters
 * for finer control over attributes
 */
static PyObject *
OpticalMedium_getthickness(OpticalMedium *self, void *closure) {
  Py_INCREF(self->_thick);
  return self->_thick;
}

static int
OpticalMedium_setthickness(OpticalMedium *self, PyObject *value, void *closure) {
  PyObject *tmp;

  if (PyFloat_AsDouble(value) < 0) {
    if (PyFloat_AsDouble(value) != -1) {
      PyErr_SetString(PyExc_ValueError, "thickness must be > 0 or -1 for infinite medium");
      return -1;
    }
  }

  tmp = self->_thick;
  Py_INCREF(value);
  self->_thick = value;
  Py_DECREF(tmp);

  return 0;
}

static PyObject *
OpticalMedium_getntype(OpticalMedium *self, void *closure) {
  Py_INCREF(self->_ntype);
  return self->_ntype;
}

static int
OpticalMedium_setntype(OpticalMedium *self, PyObject *value, void *closure) {
  PyObject *tmp;

  tmp = self->_ntype;
  Py_INCREF(value);
  self->_ntype = value;
  Py_DECREF(tmp);

  return 0;
}

static PyGetSetDef OpticalMedium_getsetters[] = {
  {
    "_thick",
    (getter) OpticalMedium_getthickness,
    (setter) OpticalMedium_setthickness,
    "thickness in nanometers, -1 or > 0",
    NULL
  },
  {
    "_ntype",
    (getter) OpticalMedium_getntype,
    (setter) OpticalMedium_setntype,
    "index type, -1 if unspecified",
    NULL
  },
  {NULL}  /* Sentinel */
};

static PyObject *
OpticalMedium_absorption_coeffs(OpticalMedium *self, PyObject *args) {
  int n_reflect;

  if (!PyArg_ParseTuple(args, "i", &n_reflect))
      return Py_None;

  PyArrayIterObject *iter1, *iter2;
  iter1 = (PyArrayIterObject *)PyArray_IterNew(self->waves);
  iter2 = (PyArrayIterObject *)PyArray_IterNew(self->nref);

  if (iter1 == NULL || iter2 == NULL) {
    return Py_None;
  }

  std::complex<double> *coeffs = new std::complex<double>[iter1->size];

  while (iter1->index < iter1->size) {
    /* calculate coefficients with iter1->dataptr and iter2->dataptr */
    double wv = *(double *)iter1->dataptr;
    std::complex<double> n_ref = *(std::complex<double> *)iter2->dataptr;

    coeffs[iter1->index] = (std::abs(n_reflect) * Py_MATH_PI * std::imag(n_ref)) / wv;

    /* increment iterator */
    PyArray_ITER_NEXT(iter1);
    PyArray_ITER_NEXT(iter2);
  }

  PyObject *out;
  int ndims = PyArray_NDIM((PyArrayObject *)self->nref);
  npy_intp *dims = PyArray_DIMS((PyArrayObject *)self->nref);
  out = PyArray_SimpleNewFromData(ndims, dims, NPY_COMPLEX128, (void *)coeffs);

  return out;
}

static PyObject *
OpticalMedium_nref_eff(OpticalMedium *self, PyObject *args) {
  PyObject *theta_obj;  // parse theta as an object
  PyObject *theta_arr;

  if (!PyArg_ParseTuple(args, "O", &theta_obj)) {
    return Py_None;
  }

  // create the iterator for nref
  PyArrayIterObject *iter, *iter_t;
  iter = (PyArrayIterObject *)PyArray_IterNew(self->nref);

  /* Handle two cases of theta --> float or array */
  double *thetas;  // placeholder for angles
  npy_intp *nref_dims = PyArray_DIMS((PyArrayObject *)self->nref);

  if (PyFloat_CheckExact(theta_obj)) {
    // check if theta is a float, make an array with theta
    // that matches shape of nref
    thetas = new double[iter->size];
    double t = PyFloat_AsDouble(theta_obj);
    for (int i = 0; i < (int)iter->size; i++) {
      thetas[i] = t;
    }

    // make an array object
    theta_arr = PyArray_SimpleNewFromData(1, nref_dims, NPY_FLOAT64, (void *)thetas);

  } else {
    // if theta is not a float, it should be an array
    // validate the size of the array matches nref
    int DTYPE = PyArray_ObjectType(theta_obj, NPY_FLOAT);
    theta_arr = PyArray_FROM_OTF(theta_obj, DTYPE, NPY_ARRAY_INOUT_ARRAY);

    // get the dimensions
    int theta_ndims = PyArray_NDIM((PyArrayObject *)theta_arr);
    npy_intp *theta_dims = PyArray_DIMS((PyArrayObject *)theta_arr);

    // validate the dimensions
    if (theta_ndims != 1) {
      PyErr_SetString(PyExc_ValueError, "theta expects float or 1-D array");
      return NULL;
    }

    if (theta_dims[0] != nref_dims[0]) {
      PyErr_SetString(PyExc_ValueError, "theta must have same number of elements as nref");
      return NULL;
    }

  }

  // set up the iterator for theta
  iter_t = (PyArrayIterObject *)PyArray_IterNew(theta_arr);

  // store results of nref_eff
  std::complex<double> *nref_eff = new std::complex<double>[iter->size];

  while (iter->index < iter->size) {
    /* calculate coefficients with iter1->dataptr and iter2->dataptr */
    double n_real = std::real(*(std::complex<double> *)iter->dataptr);
    double n_imag = std::imag(*(std::complex<double> *)iter->dataptr);
    double theta = *(double *)iter_t->dataptr;

    // compute the inner-most term of the effective index
    double in1a = pow(pow(n_imag, 2) + pow(n_real, 2), 2);
    double in1b = 2 * n_imag - n_real;
    double in1c = n_imag + n_real;
    double in1d = pow(sin(theta), 2) + pow(sin(theta), 4);
    double inner1 = in1a + in1b * in1c * in1d;

    // computer the next inner-most term
    double in2a = -pow(n_imag, 2);
    double in2b = pow(n_real, 2);
    double in2c = pow(sin(theta), 2);
    double in2d = sqrt(inner1);
    double inner2 = 0.5 * (in2a + in2b + in2c + in2d);

    // compute the outer-most term (this is the result)
    nref_eff[iter->index] = sqrt(inner2);

    /* increment iterator */
    PyArray_ITER_NEXT(iter);
  }

  PyObject *out;
  int ndims = PyArray_NDIM((PyArrayObject *)self->nref);
  npy_intp *dims = PyArray_DIMS((PyArrayObject *)self->nref);
  out = PyArray_SimpleNewFromData(ndims, dims, NPY_COMPLEX128, (void *)nref_eff);

  return out;

}

static PyObject *
OpticalMedium_admittance(OpticalMedium *self, PyObject *args) {
  OpticalMedium *inc = NULL;  // the incident medium object
  PyObject *theta_obj;  // parse theta as an object
  PyObject *theta_arr;

  if (!PyArg_ParseTuple(args, "OO", &inc, &theta_obj))
      return Py_None;

  // create a PyDict to store results
  PyObject *ret = PyDict_New();

  PyArrayIterObject *iter1, *iter2, *iter_t;
  iter1 = (PyArrayIterObject *)PyArray_IterNew(self->nref);
  iter2 = (PyArrayIterObject *)PyArray_IterNew(inc->nref);

  if (iter1 == NULL || iter2 == NULL) {
    return Py_None;
  }

  /* Handle two cases of theta --> float or array */
  double *thetas;  // placeholder for angles
  npy_intp *nref_dims = PyArray_DIMS((PyArrayObject *)self->nref);

  if (PyFloat_CheckExact(theta_obj)) {
    // check if theta is a float, make an array with theta
    // that matches shape of nref
    thetas = new double[iter1->size];
    double t = PyFloat_AsDouble(theta_obj);
    for (int i = 0; i < (int)iter1->size; i++) {
      thetas[i] = t;
    }

    // make an array object
    theta_arr = PyArray_SimpleNewFromData(1, nref_dims, NPY_FLOAT64, (void *)thetas);

  } else {
    // if theta is not a float, it should be an array
    // validate the size of the array matches nref
    int DTYPE = PyArray_ObjectType(theta_obj, NPY_FLOAT);
    theta_arr = PyArray_FROM_OTF(theta_obj, DTYPE, NPY_ARRAY_INOUT_ARRAY);

    // get the dimensions
    int theta_ndims = PyArray_NDIM((PyArrayObject *)theta_arr);
    npy_intp *theta_dims = PyArray_DIMS((PyArrayObject *)theta_arr);

    // validate the dimensions
    if (theta_ndims != 1) {
      PyErr_SetString(PyExc_ValueError, "theta expects float or 1-D array");
      return NULL;
    }

    if (theta_dims[0] != nref_dims[0]) {
      PyErr_SetString(PyExc_ValueError, "theta must have same number of elements as nref");
      return NULL;
    }

  }

  // set up the iterator for theta
  iter_t = (PyArrayIterObject *)PyArray_IterNew(theta_arr);

  std::complex<double> *admit_s = new std::complex<double>[iter1->size];
  std::complex<double> *admit_p = new std::complex<double>[iter1->size];

  while (iter1->index < iter1->size) {
    /* calculate admittances with iter1->dataptr and iter2->dataptr */
    double theta = *(double *)iter_t->dataptr;
    admit_s[iter1->index] = sqrt(
      pow(*(std::complex<double> *)iter1->dataptr, 2) - pow(*(std::complex<double> *)iter2->dataptr, 2) * sin(theta));
    admit_p[iter1->index] = pow(*(std::complex<double> *)iter1->dataptr, 2) / admit_s[iter1->index];

    /* increment iterator */
    PyArray_ITER_NEXT(iter1);
    PyArray_ITER_NEXT(iter2);
    PyArray_ITER_NEXT(iter_t);
  }

  PyObject *arr_s, *arr_p;
  int ndims = PyArray_NDIM((PyArrayObject *)self->nref);
  npy_intp *dims = PyArray_DIMS((PyArrayObject *)self->nref);
  arr_s = PyArray_SimpleNewFromData(ndims, dims, NPY_COMPLEX128, (void *)admit_s);
  arr_p = PyArray_SimpleNewFromData(ndims, dims, NPY_COMPLEX128, (void *)admit_p);

  // add objects to dict
  PyDict_SetItem(ret, Py_BuildValue("s", "s"), arr_s);
  PyDict_SetItem(ret, Py_BuildValue("s", "p"), arr_p);

  Py_INCREF(ret);
  return ret;
}

static PyObject *
OpticalMedium_admittance_eff(OpticalMedium *self, PyObject *args) {

  if (PyFloat_AsDouble(self->_thick) < 0) {
    PyErr_SetString(PyExc_ValueError, "thickness must be finite and greater than zero");
    return NULL;
  }

  OpticalMedium *inc = NULL;  // the incident medium object
  PyObject *theta_obj;  // parse theta as an object
  PyObject *theta_arr;

  if (!PyArg_ParseTuple(args, "OO", &inc, &theta_obj))
      return Py_None;

  // create a PyDict to store results
  PyObject *ret = PyDict_New();

  // build values to pass to nref_effective
  PyObject *arg = Py_BuildValue("(O)", theta_obj);
  //PyObject *keywords = PyDict_New();
  //PyDict_SetItemString(keywords, "theta", Py_True);
  PyObject *nref_eff_method = PyObject_GetAttrString((PyObject *)self, "nref_eff");

  // make call to nref_effective, clean up memory
  PyObject *nref_eff = PyObject_Call(nref_eff_method, arg, NULL);

  /* may need to check nref_eff here */

  PyArrayIterObject *iter1, *iter2, *iter_t;
  iter1 = (PyArrayIterObject *)PyArray_IterNew(nref_eff);
  iter2 = (PyArrayIterObject *)PyArray_IterNew(inc->nref);

  if (iter1 == NULL || iter2 == NULL) {
    return Py_None;
  }

  /* Handle two cases of theta --> float or array */
  double *thetas;  // placeholder for angles
  npy_intp *nref_dims = PyArray_DIMS((PyArrayObject *)self->nref);

  if (PyFloat_CheckExact(theta_obj)) {
    // check if theta is a float, make an array with theta
    // that matches shape of nref
    thetas = new double[iter1->size];
    double t = PyFloat_AsDouble(theta_obj);
    for (int i = 0; i < (int)iter1->size; i++) {
      thetas[i] = t;
    }

    // make an array object
    theta_arr = PyArray_SimpleNewFromData(1, nref_dims, NPY_FLOAT64, (void *)thetas);

  } else {
    // if theta is not a float, it should be an array
    // validate the size of the array matches nref
    int DTYPE = PyArray_ObjectType(theta_obj, NPY_FLOAT);
    theta_arr = PyArray_FROM_OTF(theta_obj, DTYPE, NPY_ARRAY_INOUT_ARRAY);

    // get the dimensions
    int theta_ndims = PyArray_NDIM((PyArrayObject *)theta_arr);
    npy_intp *theta_dims = PyArray_DIMS((PyArrayObject *)theta_arr);

    // validate the dimensions
    if (theta_ndims != 1) {
      PyErr_SetString(PyExc_ValueError, "theta expects float or 1-D array");
      return NULL;
    }

    if (theta_dims[0] != nref_dims[0]) {
      PyErr_SetString(PyExc_ValueError, "theta must have same number of elements as nref");
      return NULL;
    }

  }

  // set up the iterator for theta
  iter_t = (PyArrayIterObject *)PyArray_IterNew(theta_arr);


  std::complex<double> *admit_s = new std::complex<double>[iter1->size];
  std::complex<double> *admit_p = new std::complex<double>[iter1->size];

  while (iter1->index < iter1->size) {
    /* calculate admittances with iter1->dataptr and iter2->dataptr */
    double theta = *(double *)iter_t->dataptr;

    admit_s[iter1->index] = sqrt(
      pow(*(std::complex<double> *)iter1->dataptr, 2) - pow(*(std::complex<double> *)iter2->dataptr, 2) * sin(theta));
    admit_p[iter1->index] = pow(*(std::complex<double> *)iter1->dataptr, 2) / admit_s[iter1->index];

    /* increment iterator */
    PyArray_ITER_NEXT(iter1);
    PyArray_ITER_NEXT(iter2);
    PyArray_ITER_NEXT(iter_t);
  }

  PyObject *arr_s, *arr_p;
  int ndims = PyArray_NDIM((PyArrayObject *)self->nref);
  npy_intp *dims = PyArray_DIMS((PyArrayObject *)self->nref);
  arr_s = PyArray_SimpleNewFromData(ndims, dims, NPY_COMPLEX128, (void *)admit_s);
  arr_p = PyArray_SimpleNewFromData(ndims, dims, NPY_COMPLEX128, (void *)admit_p);

  // add objects to dict
  PyDict_SetItem(ret, Py_BuildValue("s", "s"), arr_s);
  PyDict_SetItem(ret, Py_BuildValue("s", "p"), arr_p);

  Py_INCREF(ret);

  // release result from Method Call
  Py_DECREF(nref_eff);
  Py_DECREF(arg);
  //Py_DECREF(keywords);
  Py_DECREF(nref_eff_method);

  return ret;
}

static PyObject *
OpticalMedium_path_length(OpticalMedium *self, PyObject *args) {

  if (PyFloat_AsDouble(self->_thick) < 0) {
    PyErr_SetString(PyExc_ValueError, "thickness must be finite and greater than zero");
    return NULL;
  }

  OpticalMedium *inc = NULL;  // the incident medium object
  PyObject *theta_obj;  // parse theta as an object
  PyObject *theta_arr;

  if (!PyArg_ParseTuple(args, "OO", &inc, &theta_obj))
      return Py_None;

  // build values to pass to nref_effective
  PyObject *arg = Py_BuildValue("(O)", theta_obj);
  //PyObject *keywords = PyDict_New();
  //PyDict_SetItemString(keywords, "theta", Py_True);
  PyObject *nref_eff_method = PyObject_GetAttrString((PyObject *)self, "nref_eff");

  // make call to nref_effective, clean up memory
  PyObject *nref_eff = PyObject_Call(nref_eff_method, arg, NULL);

  /* may need to check nref_eff here */

  PyArrayIterObject *iter1, *iter2, *iter_t;
  iter1 = (PyArrayIterObject *)PyArray_IterNew(inc->nref);
  iter2 = (PyArrayIterObject *)PyArray_IterNew(nref_eff);


  /* Handle two cases of theta --> float or array */
  double *thetas;  // placeholder for angles
  npy_intp *nref_dims = PyArray_DIMS((PyArrayObject *)self->nref);

  if (PyFloat_CheckExact(theta_obj)) {
    // check if theta is a float, make an array with theta
    // that matches shape of nref
    thetas = new double[iter1->size];
    double t = PyFloat_AsDouble(theta_obj);
    for (int i = 0; i < (int)iter1->size; i++) {
      thetas[i] = t;
    }

    // make an array object
    theta_arr = PyArray_SimpleNewFromData(1, nref_dims, NPY_FLOAT64, (void *)thetas);

  } else {
    // if theta is not a float, it should be an array
    // validate the size of the array matches nref
    int DTYPE = PyArray_ObjectType(theta_obj, NPY_FLOAT);
    theta_arr = PyArray_FROM_OTF(theta_obj, DTYPE, NPY_ARRAY_INOUT_ARRAY);

    // get the dimensions
    int theta_ndims = PyArray_NDIM((PyArrayObject *)theta_arr);
    npy_intp *theta_dims = PyArray_DIMS((PyArrayObject *)theta_arr);

    // validate the dimensions
    if (theta_ndims != 1) {
      PyErr_SetString(PyExc_ValueError, "theta expects float or 1-D array");
      return NULL;
    }

    if (theta_dims[0] != nref_dims[0]) {
      PyErr_SetString(PyExc_ValueError, "theta must have same number of elements as nref");
      return NULL;
    }

  }

  // set up the iterator for theta
  iter_t = (PyArrayIterObject *)PyArray_IterNew(theta_arr);


  // create output pointer array
  double *p_len = new double[iter1->size];

  while (iter1->index < iter1->size) {
    /* calculate coefficients with iter1->dataptr and iter2->dataptr */
    std::complex<double> n_inc = *(std::complex<double>  *)iter1->dataptr;
    std::complex<double>  n_eff = *(std::complex<double>  *)iter2->dataptr;
    double theta = *(double *)iter_t->dataptr;

    std::complex<double> denom = sqrt(1.0 - (pow(std::abs(n_inc), 2) * pow(sin(theta), 2) / pow(n_eff, 2)));

    p_len[iter1->index] = PyFloat_AsDouble(self->_thick) / std::real(denom);

    /* increment iterator */
    PyArray_ITER_NEXT(iter1);
    PyArray_ITER_NEXT(iter2);
    PyArray_ITER_NEXT(iter_t);
  }

  PyObject *out;
  int ndims = PyArray_NDIM((PyArrayObject *)self->nref);
  npy_intp *dims = PyArray_DIMS((PyArrayObject *)self->nref);
  out = PyArray_SimpleNewFromData(ndims, dims, NPY_FLOAT64, (void *)p_len);

  // release result from Method Call
  Py_DECREF(nref_eff);
  Py_DECREF(arg);
  //Py_DECREF(keywords);
  Py_DECREF(nref_eff_method);

  return out;

}

static PyObject *
OpticalMedium_fresnel_coeffs(OpticalMedium *self, PyObject *args) {

  if (PyFloat_AsDouble(self->_thick) < 0) {
    PyErr_SetString(PyExc_ValueError, "thickness must be finite and greater than zero");
    return NULL;
  }

  OpticalMedium *inc = NULL;  // the incident medium object
  PyObject *theta_obj;  // parse theta as an object
  PyObject *theta_arr;

  if (!PyArg_ParseTuple(args, "OO", &inc, &theta_obj))
      return Py_None;

  // create a PyDict to store results
  PyObject *ret = PyDict_New();

  PyArrayIterObject *iter1, *iter2, *iter_t;
  iter1 = (PyArrayIterObject *)PyArray_IterNew(self->nref);
  iter2 = (PyArrayIterObject *)PyArray_IterNew(inc->nref);

  if (iter1 == NULL || iter2 == NULL) {
    return Py_None;
  }


  /* Handle two cases of theta --> float or array */
  double *thetas;  // placeholder for angles
  npy_intp *nref_dims = PyArray_DIMS((PyArrayObject *)self->nref);

  if (PyFloat_CheckExact(theta_obj)) {
    // check if theta is a float, make an array with theta
    // that matches shape of nref
    thetas = new double[iter1->size];
    double t = PyFloat_AsDouble(theta_obj);
    for (int i = 0; i < (int)iter1->size; i++) {
      thetas[i] = t;
    }

    // make an array object
    theta_arr = PyArray_SimpleNewFromData(1, nref_dims, NPY_FLOAT64, (void *)thetas);

  } else {
    // if theta is not a float, it should be an array
    // validate the size of the array matches nref
    int DTYPE = PyArray_ObjectType(theta_obj, NPY_FLOAT);
    theta_arr = PyArray_FROM_OTF(theta_obj, DTYPE, NPY_ARRAY_INOUT_ARRAY);

    // get the dimensions
    int theta_ndims = PyArray_NDIM((PyArrayObject *)theta_arr);
    npy_intp *theta_dims = PyArray_DIMS((PyArrayObject *)theta_arr);

    // validate the dimensions
    if (theta_ndims != 1) {
      PyErr_SetString(PyExc_ValueError, "theta expects float or 1-D array");
      return NULL;
    }

    if (theta_dims[0] != nref_dims[0]) {
      PyErr_SetString(PyExc_ValueError, "theta must have same number of elements as nref");
      return NULL;
    }

  }

  // set up the iterator for theta
  iter_t = (PyArrayIterObject *)PyArray_IterNew(theta_arr);


  // S/P Polarized Transmission
  std::complex<double> *Ts = new std::complex<double>[iter1->size];
  std::complex<double> *Tp = new std::complex<double>[iter1->size];

  // S/P Polarized reflection
  std::complex<double> *Rs = new std::complex<double>[iter1->size];
  std::complex<double> *Rp = new std::complex<double>[iter1->size];

  // S/P Polarized Fresnel Amplitude Coefficients
  std::complex<double> *Fs = new std::complex<double>[iter1->size];
  std::complex<double> *Fp = new std::complex<double>[iter1->size];

  while (iter1->index < iter1->size) {
    /* calculate coefficients with iter1->dataptr and iter2->dataptr */
    std::complex<double> n_ref = *(std::complex<double> *)iter1->dataptr;  // self->nref
    std::complex<double> n_inc = *(std::complex<double> *)iter2->dataptr;  // inc->nref
    double theta = *(double *)iter_t->dataptr;  // theta

    // temporary variables used in calculation
    std::complex<double> ftemp;

    // this is the last term in each equation -- calculate once
    ftemp = sqrt(pow(n_ref, 2) - pow(n_inc, 2) * pow(sin(theta), 2));

    // S-polarized fresnel coefficients
    Fs[iter1->index] = (n_inc * cos(theta) - ftemp) / (n_inc * cos(theta) + ftemp);

    // P-polarized fresnel coefficients
    Fp[iter1->index] = -(pow(n_ref, 2) * cos(theta) - n_inc * ftemp) / (pow(n_ref, 2) * cos(theta) + n_inc * ftemp);

    // S and P polarized reflection
    Rs[iter1->index] = pow(std::abs(Fs[iter1->index]), 2);
    Rp[iter1->index] = pow(std::abs(Fp[iter1->index]), 2);

    // S and P Polarized transmission
    std::complex<double> one_c(1, 0);
    Ts[iter1->index] = one_c - Rs[iter1->index];
    Tp[iter1->index] = one_c - Rp[iter1->index];

    /* increment iterator */
    PyArray_ITER_NEXT(iter1);
    PyArray_ITER_NEXT(iter2);
    PyArray_ITER_NEXT(iter_t);
  }

  PyObject *arr_Ts, *arr_Tp, *arr_Rs, *arr_Rp, *arr_Fs, *arr_Fp;
  int ndims = PyArray_NDIM((PyArrayObject *)self->nref);
  npy_intp *dims = PyArray_DIMS((PyArrayObject *)self->nref);
  arr_Ts = PyArray_SimpleNewFromData(ndims, dims, NPY_COMPLEX128, (void *)Ts);
  arr_Tp = PyArray_SimpleNewFromData(ndims, dims, NPY_COMPLEX128, (void *)Tp);
  arr_Rs = PyArray_SimpleNewFromData(ndims, dims, NPY_COMPLEX128, (void *)Rs);
  arr_Rp = PyArray_SimpleNewFromData(ndims, dims, NPY_COMPLEX128, (void *)Rp);
  arr_Fs = PyArray_SimpleNewFromData(ndims, dims, NPY_COMPLEX128, (void *)Fs);
  arr_Fp = PyArray_SimpleNewFromData(ndims, dims, NPY_COMPLEX128, (void *)Fp);

  // add objects to dict
  PyDict_SetItem(ret, Py_BuildValue("s", "Ts"), arr_Ts);
  PyDict_SetItem(ret, Py_BuildValue("s", "Tp"), arr_Tp);
  PyDict_SetItem(ret, Py_BuildValue("s", "Rs"), arr_Rs);
  PyDict_SetItem(ret, Py_BuildValue("s", "Rp"), arr_Rp);
  PyDict_SetItem(ret, Py_BuildValue("s", "Fs"), arr_Fs);
  PyDict_SetItem(ret, Py_BuildValue("s", "Fp"), arr_Fp);

  Py_INCREF(ret);

  return ret;
}


static PyMethodDef OpticalMedium_methods[] = {
    {"absorption_coeffs", (PyCFunction) OpticalMedium_absorption_coeffs, METH_VARARGS,
     "Calculates the absorption coefficient for n_ref reflections"
    },
    {"nref_eff", (PyCFunction) OpticalMedium_nref_eff, METH_VARARGS,
     "Calculates the effective refractive index through the medium"
    },
    {"admittance", (PyCFunction) OpticalMedium_admittance, METH_VARARGS,
     "Return the admittances between self and incident medium"
    },
    {"admittance_eff", (PyCFunction) OpticalMedium_admittance_eff, METH_VARARGS,
     "Return the effective admittances between medium and incident medium"
    },
    {"path_length", (PyCFunction) OpticalMedium_path_length, METH_VARARGS,
     "Calculates the estimated optical path length through the medium"
    },
    {"fresnel_coeffs", (PyCFunction) OpticalMedium_fresnel_coeffs, METH_VARARGS,
     "Calculates the fresnel amplitudes & intensities of the medium"
    },
    {"__getstate__", (PyCFunction) OpticalMedium_getstate, METH_NOARGS,
      "OpticalMedium the Custom object"
    },
    {"__setstate__", (PyCFunction) OpticalMedium_setstate, METH_O,
      "Un-pickle the OpticalMedium object"
    },
    {NULL}  /* Sentinel */
};

// c++ does not technically support designated initializers
// this style does not work with windows c++ compiler
//static PyTypeObject OpticalMediumType = {
//    PyVarObject_HEAD_INIT(NULL, 0)
//    .tp_name = "medium.OpticalMedium",
//    .tp_doc = PyDoc_STR("cpp optical medium"),
//    .tp_basicsize = sizeof(OpticalMedium),
//    .tp_itemsize = 0,
//    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
//    .tp_new = OpticalMedium_new,
//    .tp_init = (initproc) OpticalMedium_init,
//    .tp_dealloc = (destructor) OpticalMedium_dealloc,
//    .tp_members = OpticalMedium_members
//    //.tp_methods = Custom_methods,
//};


// all initializers must be declared for c++
static PyTypeObject OpticalMediumType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "medium.OpticalMedium",                           /* tp_name */
    sizeof(OpticalMedium),                            /* tp_basicsize */
    0,                                                /* tp_itemsize */
    (destructor)OpticalMedium_dealloc,                /* tp_dealloc */
    0, // offsetof(PyTypeObject, tp_vectorcall),      /* tp_vectorcall_offset */
    0,                                                /* tp_getattr */
    0,                                                /* tp_setattr */
    0,                                                /* tp_as_async */
    0, // (reprfunc)type_repr,                        /* tp_repr */
    0, // &type_as_number,                            /* tp_as_number */
    0,                                                /* tp_as_sequence */
    0,                                                /* tp_as_mapping */
    0,                                                /* tp_hash */
    0, // (ternaryfunc)type_call,                     /* tp_call */
    0,                                                /* tp_str */
    0, // (getattrofunc)type_getattro,                /* tp_getattro */
    0, // (setattrofunc)type_setattro,                /* tp_setattro */
    0,                                                /* tp_as_buffer */
                                                      /* tp_flags */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_BASETYPE,
    /*| Py_TPFLAGS_TYPE_SUBCLASS | Py_TPFLAGS_HAVE_VECTORCALL */
    PyDoc_STR("cpp optical medium"),                  /* tp_doc */
    (traverseproc)OpticalMedium_traverse,             /* tp_traverse */
    (inquiry)OpticalMedium_clear,                     /* tp_clear */
    0,                                                /* tp_richcompare */
    0, // offsetof(PyTypeObject, tp_weaklist),        /* tp_weaklistoffset */
    0,                                                /* tp_iter */
    0,                                                /* tp_iternext */
    OpticalMedium_methods,                            /* tp_methods */
    OpticalMedium_members,                            /* tp_members */
    OpticalMedium_getsetters,                         /* tp_getset */
    0,                                                /* tp_base */
    0,                                                /* tp_dict */
    0,                                                /* tp_descr_get */
    0,                                                /* tp_descr_set */
    0, // offsetof(PyTypeObject, tp_dict),            /* tp_dictoffset */
    (initproc)OpticalMedium_init,                     /* tp_init */
    0,                                                /* tp_alloc */
    OpticalMedium_new,                                /* tp_new */
    0, // PyObject_GC_Del,                            /* tp_free */
    0 // (inquiry)type_is_gc,                         /* tp_is_gc */
};

static PyModuleDef mediummodule = {
  PyModuleDef_HEAD_INIT,  /* m_base */
  "OpticalMedium",        /* m_name */
  "CPP optical medium",   /* m_doc */
  -1,                     /* m_size */
  NULL
};

PyMODINIT_FUNC
PyInit_medium(void) {
  PyObject *m;
  if (PyType_Ready(&OpticalMediumType) < 0)
    return NULL;

  // import numpy modules
  import_array();

  m = PyModule_Create(&mediummodule);
  if (m == NULL)
    return NULL;

  Py_INCREF(&OpticalMediumType);
  if (PyModule_AddObject(m, "OpticalMedium", (PyObject *) &OpticalMediumType) < 0) {
    Py_DECREF(&OpticalMediumType);
    Py_DECREF(m);
    return NULL;
  }

  return m;
}

