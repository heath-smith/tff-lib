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
  PyObject *wavelengths;
  PyObject *ref_index;
  double thickness;
  PyObject *material;
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
    Py_VISIT(self->wavelengths);
    Py_VISIT(self->ref_index);
    Py_VISIT(self->material);
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
    Py_CLEAR(self->wavelengths);
    Py_CLEAR(self->ref_index);
    Py_CLEAR(self->material);
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
    self->wavelengths = NULL;
    self->ref_index = NULL;
    self->thickness = -1;
    self->material = PyUnicode_FromString("");  // initialize empty string
    if (self->material == NULL) {
      Py_DECREF(self);
    }
  }
  return (PyObject *) self;
}

/* Define the initialization function, __init__()
 */
static int
OpticalMedium_init(OpticalMedium *self, PyObject *args, PyObject *kwds) {
  static char *kwlist[] = {
    "wavelengths", "ref_index", "thickness", "material", NULL };
  PyObject *wavelengths = NULL, *ref_index = NULL, *material = NULL, *tmp;
  PyObject *wv_arr = NULL, *ref_arr = NULL;
  int DTYPE, iscomplex;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOdO", kwlist,
                                    &wavelengths, &ref_index,
                                    &self->thickness, &material))
    return -1;


  if (wavelengths) {

    DTYPE = PyArray_ObjectType(wavelengths, NPY_FLOAT);
    iscomplex = PyTypeNum_ISCOMPLEX(DTYPE);
    wv_arr = PyArray_FROM_OTF(wavelengths, DTYPE, NPY_ARRAY_INOUT_ARRAY);

    if (wv_arr == NULL) {
      Py_CLEAR(wv_arr);
      return -1;
    }

    if (PyArray_NDIM((PyArrayObject *)wv_arr) != 1) {
      Py_CLEAR(wv_arr);
      PyErr_SetString(PyExc_ValueError, "wavelengths expects 1-dimensional array");
      return -1;
    }

  }

  if (ref_index) {

    DTYPE = PyArray_ObjectType(ref_index, NPY_FLOAT);
    iscomplex = PyTypeNum_ISCOMPLEX(DTYPE);
    ref_arr = PyArray_FROM_OTF(ref_index, DTYPE, NPY_ARRAY_INOUT_ARRAY);

    if (ref_arr == NULL) {
      Py_CLEAR(ref_arr);
      return -1;
    }

    if (PyArray_NDIM((PyArrayObject *)ref_arr) != 1) {
      Py_CLEAR(ref_arr);
      PyErr_SetString(PyExc_ValueError, "ref_index expects 1-dimensional array");
      return -1;
    }

  }

  // verify both array's have same number of elements
  if (PyArray_SIZE((PyArrayObject *)wv_arr) != PyArray_SIZE((PyArrayObject *)ref_arr)) {
    Py_CLEAR(wv_arr);
    Py_CLEAR(ref_arr);
    PyErr_SetString(
      PyExc_ValueError, "wavelengths and ref_index must have the same number of elements");
    return -1;
  }

  self->wavelengths = Py_BuildValue("N", wv_arr);
  self->ref_index = Py_BuildValue("N", ref_arr);

  Py_INCREF(self->wavelengths);
  Py_INCREF(self->ref_index);

  Py_XDECREF(wv_arr);
  Py_XDECREF(ref_arr);


  if (material) {
    tmp = self->material;
    Py_INCREF(material);
    self->material = material;
    Py_XDECREF(tmp);
  }

  return 0;
}

static PyMemberDef OpticalMedium_members[] = {
  {"wavelengths", T_OBJECT_EX, offsetof(OpticalMedium, wavelengths), 0, "wavelengths"},
  {"ref_index", T_OBJECT_EX, offsetof(OpticalMedium, ref_index), 0, "ref_index"},
  {"thickness", T_DOUBLE, offsetof(OpticalMedium, thickness), 0, "thickness"},
  {"material", T_OBJECT_EX, offsetof(OpticalMedium, material), 0, "material"},
  {NULL}  /* Sentinel */
};

/* Provide customized getters/setters
 * for finer control over attributes
 */
static int
OpticalMedium_getthickness(OpticalMedium *self, void *closure) {
  return self->thickness;
}

static int
OpticalMedium_setthickness(OpticalMedium *self, PyObject *value, void *closure) {
  double tmp = PyFloat_AsDouble(value);

  if (tmp < 0) {
    if (tmp != -1) {
      PyErr_SetString(PyExc_ValueError, "thickness must be > 0 or -1 for inf.");
      return -1;
    }
  }

  self->thickness = tmp;
  return 0;
}

static PyGetSetDef OpticalMedium_getsetters[] = {
  {
    "thickness",
    (getter) OpticalMedium_getthickness,
    (setter) OpticalMedium_setthickness,
    "thickness in nanometers",
    NULL
  },
  {NULL}  /* Sentinel */
};

static PyObject *
OpticalMedium_admittance(OpticalMedium *self, PyObject *args) {
  OpticalMedium *inc = NULL;  // the incident medium object
  double theta = 0.0;

  if (! PyArg_ParseTuple(args, "Od", &inc, &theta))
      return Py_None;

  // create a PyDict to store results
  PyObject *ret = PyDict_New();

  PyArrayIterObject *iter1, *iter2;
  iter1 = (PyArrayIterObject *)PyArray_IterNew(self->ref_index);
  iter2 = (PyArrayIterObject *)PyArray_IterNew(inc->ref_index);

  if (iter1 == NULL || iter2 == NULL) {
    return Py_None;
  }

  std::complex<double> *admit_s = new std::complex<double>[iter1->size];
  std::complex<double> *admit_p = new std::complex<double>[iter1->size];

  while (iter1->index < iter1->size) {
    /* calculate admittances with iter1->dataptr and iter2->dataptr */
    admit_s[iter1->index] = sqrt(
      pow(*(double *)iter1->dataptr, 2) - pow(*(double *)iter2->dataptr, 2) * sin(theta));
    admit_p[iter1->index] = pow(*(double *)iter1->dataptr, 2) / admit_s[iter1->index];

    /* increment iterator */
    PyArray_ITER_NEXT(iter1);
    PyArray_ITER_NEXT(iter2);
  }

  PyObject *arr_s, *arr_p;
  int ndims = PyArray_NDIM((PyArrayObject *)self->ref_index);
  npy_intp *dims = PyArray_DIMS((PyArrayObject *)self->ref_index);
  arr_s = PyArray_SimpleNewFromData(ndims, dims, NPY_COMPLEX128, (void *)admit_s);
  arr_p = PyArray_SimpleNewFromData(ndims, dims, NPY_COMPLEX128, (void *)admit_p);

  // add objects to dict
  PyDict_SetItem(ret, Py_BuildValue("s", "s"), arr_s);
  PyDict_SetItem(ret, Py_BuildValue("s", "p"), arr_p);

  Py_INCREF(ret);
  return ret;
}

// std::map<char, std::complex<double> *> OpticalMedium::admittance(
//                                           const OpticalMedium &inc_medium,
//                                           double theta) const {
//   // create new admittance map
//   std::map <char, std::complex<double> *> admit;
//   admit['s'] = new std::complex<double>[n_];
//   admit['p'] = new std::complex<double>[n_];
//
//   for (size_t i = 0; i < n_; i++) {
//     admit['s'][i] = sqrt(
//       pow(ref_index[i], 2) - pow(inc_medium.ref_index[i], 2) * sin(theta));
//     admit['p'][i] = pow(ref_index[i], 2) / admit['s'][i];
//   }
//   return admit;
// }

static PyMethodDef OpticalMedium_methods[] = {
    {"admittance", (PyCFunction) OpticalMedium_admittance, METH_VARARGS,
     "Return the admittances between self and incident medium"
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

