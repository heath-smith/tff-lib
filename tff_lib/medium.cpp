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

    DTYPE = PyArray_ObjectType(waves, NPY_FLOAT);
    temp_wv = PyArray_FROM_OTF(waves, DTYPE, NPY_ARRAY_INOUT_ARRAY);

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

    DTYPE = PyArray_ObjectType(nref, NPY_FLOAT);
    temp_nref = PyArray_FROM_OTF(nref, DTYPE, NPY_ARRAY_INOUT_ARRAY);

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

    if (
      PyLong_AsLong(ntype) != 0 && PyLong_AsLong(ntype) != -1 && PyLong_AsLong(ntype) != 1) {
        PyErr_SetString(PyExc_ValueError, "ntype must be 1, 0, or -1");
        return -1;
    }

    tmp = self->_ntype;
    Py_INCREF(ntype);
    self->_ntype = ntype;
    Py_XDECREF(tmp);
  }

  return 0;
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

    if (
      PyFloat_AsDouble(value) != 0 && PyFloat_AsDouble(value) != -1 && PyFloat_AsDouble(value) != 1) {
        PyErr_SetString(PyExc_ValueError, "ntype must be 1, 0, or -1");
        return -1;
    }

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
    "index type, 1, 0, or -1",
    NULL
  },
  {NULL}  /* Sentinel */
};

static PyObject *
OpticalMedium_admittance(OpticalMedium *self, PyObject *args) {
  OpticalMedium *inc = NULL;  // the incident medium object
  double theta = 0.0;

  if (!PyArg_ParseTuple(args, "Od", &inc, &theta))
      return Py_None;

  // create a PyDict to store results
  PyObject *ret = PyDict_New();

  PyArrayIterObject *iter1, *iter2;
  iter1 = (PyArrayIterObject *)PyArray_IterNew(self->nref);
  iter2 = (PyArrayIterObject *)PyArray_IterNew(inc->nref);

  if (iter1 == NULL || iter2 == NULL) {
    return Py_None;
  }

  std::complex<double> *admit_s = new std::complex<double>[iter1->size];
  std::complex<double> *admit_p = new std::complex<double>[iter1->size];

  while (iter1->index < iter1->size) {
    /* calculate admittances with iter1->dataptr and iter2->dataptr */
    admit_s[iter1->index] = sqrt(
      pow(*(std::complex<double> *)iter1->dataptr, 2) - pow(*(std::complex<double> *)iter2->dataptr, 2) * sin(theta));
    admit_p[iter1->index] = pow(*(std::complex<double> *)iter1->dataptr, 2) / admit_s[iter1->index];

    /* increment iterator */
    PyArray_ITER_NEXT(iter1);
    PyArray_ITER_NEXT(iter2);
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
    coeffs[iter1->index] = (
      n_reflect * Py_MATH_PI * std::imag(*(std::complex<double> *)iter2->dataptr)) / *(double *)iter2->dataptr;

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
OpticalMedium_nref_effective(OpticalMedium *self, PyObject *args) {
  double theta = 0.0;

  if (!PyArg_ParseTuple(args, "d", &theta)) {
    return Py_None;
  }

  PyArrayIterObject *iter;
  iter = (PyArrayIterObject *)PyArray_IterNew(self->nref);

  std::complex<double> *nref_eff = new std::complex<double>[iter->size];

  while (iter->index < iter->size) {
    /* calculate coefficients with iter1->dataptr and iter2->dataptr */
    double n_real = std::real(*(std::complex<double> *)iter->dataptr);
    double n_imag = std::imag(*(std::complex<double> *)iter->dataptr);

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
OpticalMedium_path_length(OpticalMedium *self, PyObject *args) {

  if (PyFloat_AsDouble(self->_thick) < 0) {
    PyErr_SetString(PyExc_ValueError, "thickness must be finite and greater than zero");
    return NULL;
  }

  OpticalMedium *inc = NULL;  // the incident medium object
  double theta = 0.0;

  if (!PyArg_ParseTuple(args, "Od", &inc, &theta))
      return Py_None;

  // build values to pass to nref_effective
  PyObject *arg = Py_BuildValue("(d)", theta);
  //PyObject *keywords = PyDict_New();
  //PyDict_SetItemString(keywords, "theta", Py_True);
  PyObject *nref_eff_method = PyObject_GetAttrString((PyObject *)self, "nref_effective");

  // make call to nref_effective, clean up memory
  PyObject *nref_eff = PyObject_Call(nref_eff_method, arg, NULL);

  /* may need to check nref_eff here */

  PyArrayIterObject *iter1, *iter2;
  iter1 = (PyArrayIterObject *)PyArray_IterNew(inc->nref);
  iter2 = (PyArrayIterObject *)PyArray_IterNew(nref_eff);

  // create output pointer array
  double *p_len = new double[iter1->size];

  while (iter1->index < iter1->size) {
    /* calculate coefficients with iter1->dataptr and iter2->dataptr */
    std::complex<double> n_inc = *(std::complex<double>  *)iter1->dataptr;
    std::complex<double>  n_eff = *(std::complex<double>  *)iter2->dataptr;

    std::complex<double> denom = sqrt(1.0 - (pow(std::abs(n_inc), 2) * pow(sin(theta), 2) / pow(n_eff, 2)));

    p_len[iter1->index] = PyFloat_AsDouble(self->_thick) / std::real(denom);

    /* increment iterator */
    PyArray_ITER_NEXT(iter1);
    PyArray_ITER_NEXT(iter2);
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
OpticalMedium_admit_effective(OpticalMedium *self, PyObject *args) {

  if (PyFloat_AsDouble(self->_thick) < 0) {
    PyErr_SetString(PyExc_ValueError, "thickness must be finite and greater than zero");
    return NULL;
  }

  OpticalMedium *inc = NULL;  // the incident medium object
  double theta = 0.0;

  if (!PyArg_ParseTuple(args, "Od", &inc, &theta))
      return Py_None;

  // create a PyDict to store results
  PyObject *ret = PyDict_New();

  // build values to pass to nref_effective
  PyObject *arg = Py_BuildValue("(d)", theta);
  //PyObject *keywords = PyDict_New();
  //PyDict_SetItemString(keywords, "theta", Py_True);
  PyObject *nref_eff_method = PyObject_GetAttrString((PyObject *)self, "nref_effective");

  // make call to nref_effective, clean up memory
  PyObject *nref_eff = PyObject_Call(nref_eff_method, arg, NULL);

  /* may need to check nref_eff here */

  PyArrayIterObject *iter1, *iter2;
  iter1 = (PyArrayIterObject *)PyArray_IterNew(nref_eff);
  iter2 = (PyArrayIterObject *)PyArray_IterNew(inc->nref);

  if (iter1 == NULL || iter2 == NULL) {
    return Py_None;
  }

  std::complex<double> *admit_s = new std::complex<double>[iter1->size];
  std::complex<double> *admit_p = new std::complex<double>[iter1->size];

  while (iter1->index < iter1->size) {
    /* calculate admittances with iter1->dataptr and iter2->dataptr */
    admit_s[iter1->index] = sqrt(
      pow(*(std::complex<double> *)iter1->dataptr, 2) - pow(*(std::complex<double> *)iter2->dataptr, 2) * sin(theta));
    admit_p[iter1->index] = pow(*(std::complex<double> *)iter1->dataptr, 2) / admit_s[iter1->index];

    /* increment iterator */
    PyArray_ITER_NEXT(iter1);
    PyArray_ITER_NEXT(iter2);
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
OpticalMedium_fresnel_coeffs(OpticalMedium *self, PyObject *args) {

  if (PyFloat_AsDouble(self->_thick) < 0) {
    PyErr_SetString(PyExc_ValueError, "thickness must be finite and greater than zero");
    return NULL;
  }

  OpticalMedium *inc = NULL;  // the incident medium object
  double theta = 0.0;

  if (!PyArg_ParseTuple(args, "Od", &inc, &theta))
      return Py_None;

  // create a PyDict to store results
  PyObject *ret = PyDict_New();


  PyArrayIterObject *iter1, *iter2;
  iter1 = (PyArrayIterObject *)PyArray_IterNew(self->nref);
  iter2 = (PyArrayIterObject *)PyArray_IterNew(inc->nref);

  if (iter1 == NULL || iter2 == NULL) {
    return Py_None;
  }

  std::complex<double> *Ts = new std::complex<double>[iter1->size];
  std::complex<double> *Tp = new std::complex<double>[iter1->size];

  while (iter1->index < iter1->size) {
    /* calculate admittances with iter1->dataptr and iter2->dataptr */
    admit_s[iter1->index] = sqrt(
      pow(*(std::complex<double> *)iter1->dataptr, 2) - pow(*(std::complex<double> *)iter2->dataptr, 2) * sin(theta));
    admit_p[iter1->index] = pow(*(std::complex<double> *)iter1->dataptr, 2) / admit_s[iter1->index];

    /* increment iterator */
    PyArray_ITER_NEXT(iter1);
    PyArray_ITER_NEXT(iter2);
  }

  PyObject *arr_Ts, *arr_Tp;
  int ndims = PyArray_NDIM((PyArrayObject *)self->nref);
  npy_intp *dims = PyArray_DIMS((PyArrayObject *)self->nref);
  arr_Ts = PyArray_SimpleNewFromData(ndims, dims, NPY_COMPLEX128, (void *)Ts);
  arr_Tp = PyArray_SimpleNewFromData(ndims, dims, NPY_COMPLEX128, (void *)Tp);

  // add objects to dict
  PyDict_SetItem(ret, Py_BuildValue("s", "Ts"), arr_Ts);
  PyDict_SetItem(ret, Py_BuildValue("s", "Tp"), arr_Tp);
  PyDict_SetItem(ret, Py_BuildValue("s", "Rs"), arr_Rs);
  PyDict_SetItem(ret, Py_BuildValue("s", "Rp"), arr_Rp);
  PyDict_SetItem(ret, Py_BuildValue("s", "rs"), arr_rs);
  PyDict_SetItem(ret, Py_BuildValue("s", "rp"), arr_rp);

  Py_INCREF(ret);


  return ret;
}



static PyMethodDef OpticalMedium_methods[] = {
    {"admittance", (PyCFunction) OpticalMedium_admittance, METH_VARARGS,
     "Return the admittances between self and incident medium"
    },
    {"absorption_coeffs", (PyCFunction) OpticalMedium_absorption_coeffs, METH_VARARGS,
     "Calculates the absorption coefficient for n_ref reflections"
    },
    {"nref_effective", (PyCFunction) OpticalMedium_nref_effective, METH_VARARGS,
     "Calculates the effective refractive index through the medium"
    },
    {"path_length", (PyCFunction) OpticalMedium_path_length, METH_VARARGS,
     "Calculates the estimated optical path length through the medium"
    },
    {"admit_effective", (PyCFunction) OpticalMedium_admit_effective, METH_VARARGS,
     "Return the effective admittances between medium and incident medium"
    },
    {"fresnel_coeffs", (PyCFunction) OpticalMedium_fresnel_coeffs, METH_VARARGS,
     "Calculates the fresnel amplitudes & intensities of the medium"
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

