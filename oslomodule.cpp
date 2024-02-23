#include <vector>
#include <iostream>
#include <random>

#include <python3.6/Python.h>
#include <python3.6/numpy/arrayobject.h>

class BaseIterator {
    /**
     * Class BaseIterator
     *
     * This class forms the basis for the iterators.
     */
private:
    double probability;

protected:
    unsigned length;

    unsigned height;
    unsigned avlch_size;

    unsigned steps;
    unsigned crossover_step;

    std::vector<unsigned> slopes;
    std::vector<unsigned> heights;
    std::vector<unsigned> avlch_sizes;
    std::vector<unsigned> thresholds;

    std::mt19937 generator;
    std::discrete_distribution<> base_distribution;

    BaseIterator (unsigned length, double probability);

    inline int distribution();

    virtual void next() = 0;

public:
    bool last_site_relaxed;

    std::vector<unsigned> get_slopes() {
        return slopes;
    }

    void store_height();
    std::vector<unsigned> get_heights() {
        return heights;
    }

    void store_avlch_size();
    std::vector<unsigned> get_avlch_sizes() {
        return avlch_sizes;
    }

    unsigned get_crossover_step(){
        return crossover_step;
    }
};

inline int BaseIterator::distribution(){
    /**
     * function distribution
     * Returns 1 or 2 with probability and 1 - probability.
     *
     * Returns:
     *  int: Either 1 or 2.
     */
    return base_distribution(generator) + 1;
}

void BaseIterator::store_height(){
    heights.push_back(height);
}

void BaseIterator::store_avlch_size(){
    avlch_sizes.push_back(avlch_size);
}

BaseIterator::BaseIterator(unsigned length, double probability)
: probability(probability),
  length(length),
  height(0),
  avlch_size(0),
  steps(0),
  crossover_step(0),
  slopes(length, 0),
  heights(0),
  avlch_sizes(0),
  thresholds(length),
  generator(std::random_device{}()),
  base_distribution({probability, 1 - probability}),
  last_site_relaxed(false) {
    /**
     * function BaseIterator
     * Class constructor for BaseIterator.
     *
     * Parameters:
     *   int length: The number of sites in the system.
     *   double probability: The probability with which a threshold value of 1
     *      will be chosen.
     *
     * Errors:
     *   Throws an error if probability is less than 0 or greater than 1.
     */

    try {
        if (probability < 0 || probability > 1) {
            throw 1;
        }
    }
    catch (int e) {
        PyErr_SetString(PyExc_ValueError, "BaseIterator constructor: probability must be between 0 and 1.");
    }

    for (unsigned i = 0; i < length; ++i) {
        thresholds[i] = distribution();
    }
}



class NaiveOslo : public BaseIterator {
    /**
     * This class implements a naive algorithm for the model.
     */

public:
    NaiveOslo(unsigned length, double probability);
    void next();
};

NaiveOslo::NaiveOslo(unsigned length, double probability)
: BaseIterator(length, probability) {
    /**
     * function NaiveOslo
     * Class constructor for NaiveOslo
     * Parameters:
     *  int length: The number of sites in the system.
     *  double probability: The probability with which a threshold value of 1
     *      will be chosen.
     */
}

void NaiveOslo::next() {
    /**
     * function next
     * Runs a naive algorithm until a stable state is reached.
     */

    avlch_size = 0;

    ++slopes[0];
    ++height;
    bool stable = true;
    do {
        stable = true;

        if (slopes[0] > thresholds[0]) {
            ++avlch_size;

            slopes[0] -= 2;
            --height;
            slopes[1] += 1;
            thresholds[0] = distribution();
            stable = false;
        }

        for (unsigned i = 1; i < length - 1; ++i) {
            if (slopes[i] > thresholds[i]) {
                ++avlch_size;
                slopes[i - 1] += 1;
                slopes[i] -= 2;
                slopes[i + 1] += 1;
                thresholds[i] = distribution();
                stable = false;
            }
        }

        if (slopes[length - 1] > thresholds[length - 1]) {
            ++avlch_size;
            slopes[length - 1] -= 1;
            slopes[length - 2] += 1;
            thresholds[length - 1] = distribution();

            if (!last_site_relaxed) {
                crossover_step = steps;
                last_site_relaxed = true;
            }

            stable = false;
        }
    }
    while (!stable);

    ++steps;

}

class EfficientOslo : public BaseIterator {
    /**
     * This class implements a more efficient algorithm for the model.
     */
private:
    void crit_check(unsigned site, std::vector<unsigned> & just_critical);

public:
    EfficientOslo(unsigned length, double probability);
    void next();
};

EfficientOslo::EfficientOslo(unsigned length, double probability)
: BaseIterator(length, probability) {
    /**
     * function EfficientOslo
     * Class constructor for EfficientOslo
     * Parameters:
     *  int length: The number of sites in the system.
     *  double probability: The probability with which a threshold value of 1
     *      will be chosen.
     */
}

void EfficientOslo::crit_check(unsigned site, std::vector<unsigned> & just_critical) {
    /**
     * function crit_check
     * Checks if a site has just become critical and changes the just_critical
     * vector.
     *
     * Parameters:
     *  unsigned site: The position of the site to check.
     *  std::vector<unsigned> & just_critical: A reference to the just_critical
     *      vector.
     */

    if (slopes[site] == thresholds[site] + 1) {
        just_critical.push_back(site);
    }
}

void EfficientOslo::next() {
    /**
     * function next
     * Runs the more efficient algorithm until a stable state is reached.
     */

    ++slopes[0];
    ++height;

    avlch_size = 0;

    if (slopes[0] > thresholds[0]) {
        std::vector<unsigned> just_critical({0});

        while (!just_critical.empty()) {
            std::vector<unsigned> new_just_critical(0);

            for (unsigned i = 0; i < just_critical.size(); ++i) {

                unsigned position = just_critical[i];

                if (slopes[position] > thresholds[position]) {

                    ++avlch_size;

                    if (position == 0) {
                        slopes[0] -= 2;
                        --height;
                        slopes[1] += 1;

                        thresholds[0] = distribution();

                        crit_check(0, new_just_critical);
                        crit_check(1, new_just_critical);

                    } else if (position == length -1) {
                        slopes[length - 2] += 1;
                        slopes[length - 1] -= 1;

                        thresholds[length - 1] = distribution();

                        if (!last_site_relaxed) {
                            crossover_step = steps;
                            last_site_relaxed = true;
                        }

                        crit_check(length - 1, new_just_critical);
                        crit_check(length - 2, new_just_critical);

                    } else {
                        slopes[position - 1] += 1;
                        slopes[position] -= 2;
                        slopes[position + 1] += 1;

                        thresholds[position] = distribution();

                        crit_check(position, new_just_critical);
                        crit_check(position - 1, new_just_critical);
                        crit_check(position + 1, new_just_critical);
                    }
                }
            }

            just_critical = new_just_critical;

        }

    }

    ++steps;
}

struct output {
    /**
     * struct output
     * A struct to store the output of the Newton-Cotes integration algorithms.
     *
     * Members:
     *   std::vector<unsigned> slopes: A vector containing the slopes at the end of a simulation.
     *   std::vector<unsigned> heights: A vector containing the heights at each time point.
     *   std::vector<unsigned> avlch_sizes: A vector containing the avalanche sizes at each point.
     *   unsigned crossover_step: The step at which the first recurrent state occurred.
     */

    std::vector<unsigned> slopes;
    std::vector<unsigned> heights;
    std::vector<unsigned> avlch_sizes;
    unsigned crossover_step;
};

bool check_slopes(std::vector<unsigned> slopes, std::vector<unsigned> thresholds)

output oslo_efficient(unsigned length, double probability, unsigned iterations) {
    EfficientOslo iterator(length, probability);

    for (unsigned i = 0; i < iterations; ++i) {
        iterator.store_height();
        iterator.store_avlch_size();
        iterator.next();
    }

    output output_struct;

    output_struct.slopes = iterator.get_slopes();
    output_struct.heights = iterator.get_heights();
    output_struct.avlch_sizes = iterator.get_avlch_sizes();
    output_struct.crossover_step = iterator.get_crossover_step();

    return output_struct;
}

output oslo_naive(unsigned length, double probability, unsigned iterations) {
    NaiveOslo iterator(length, probability);

    for (unsigned i = 0; i < iterations; ++i) {
        iterator.store_height();
        iterator.store_avlch_size();
        iterator.next();
    }

    output output_struct;

    output_struct.slopes = iterator.get_slopes();
    output_struct.heights = iterator.get_heights();
    output_struct.avlch_sizes = iterator.get_avlch_sizes();
    output_struct.crossover_step = iterator.get_crossover_step();

    return output_struct;
}

template <typename T> void
vector_to_ndarray (std::vector<T> &vector, PyObject* &ndarray) {
    for (unsigned i = 0; i < vector.size(); ++i) {
        npy_intp index = i;
        uint64_t * ptr = (uint64_t *)PyArray_GetPtr((PyArrayObject*)ndarray, &index);
        * ptr = vector[i];
    }
}

static PyObject * output_struct_to_dict(output output_struct) {

    npy_intp slopes_size = output_struct.slopes.size();
    PyObject * slopes = (PyObject*)PyArray_SimpleNew(1, &slopes_size, NPY_UINT64);
    vector_to_ndarray<unsigned>(output_struct.slopes, slopes);

    npy_intp heights_size = output_struct.heights.size();
    PyObject * heights = (PyObject*)PyArray_SimpleNew(1, &heights_size, NPY_UINT64);
    vector_to_ndarray<unsigned>(output_struct.heights, heights);

    npy_intp avlch_sizes_size = output_struct.avlch_sizes.size();
    PyObject * avlch_sizes = (PyObject*)PyArray_SimpleNew(1, &avlch_sizes_size, NPY_UINT64);
    vector_to_ndarray<unsigned>(output_struct.avlch_sizes, avlch_sizes);

    PyObject * crossover_step = PyLong_FromUnsignedLong(output_struct.crossover_step);

    PyObject* output_dict = PyDict_New();
    Py_INCREF(output_dict);

    PyDict_SetItemString(output_dict, "slopes", slopes);
    PyDict_SetItemString(output_dict, "heights", heights);
    PyDict_SetItemString(output_dict, "avlch_sizes", avlch_sizes);
    PyDict_SetItemString(output_dict, "crossover_step", crossover_step);

    Py_DECREF(heights);
    Py_DECREF(slopes);
    Py_DECREF(avlch_sizes);
    Py_DECREF(crossover_step);

    return output_dict;
}



static PyObject*
oslo_efficient_python (PyObject *self, PyObject *args) {

    unsigned length;
    double probability;
    unsigned iterations;

    if (!PyArg_ParseTuple(args, "IdI", &length, & probability, &iterations)) {
        return NULL;
    }

    output output_struct = oslo_efficient(length, probability, iterations);

    return output_struct_to_dict(output_struct);

}

static PyObject*
oslo_naive_python (PyObject *self, PyObject *args) {

    unsigned length;
    double probability;
    unsigned iterations;

    if (!PyArg_ParseTuple(args, "IdI", &length, & probability, &iterations)) {
        return NULL;
    }

    output output_struct = oslo_naive(length, probability, iterations);

    return output_struct_to_dict(output_struct);

}

static PyMethodDef methods[] = {
    {"oslo_efficient", oslo_efficient_python, METH_VARARGS,
    "Runs the efficient algorithm for the Oslo model."},
    {"oslo_naive", oslo_naive_python, METH_VARARGS,
    "Runs the naive algorithm for the Oslo model."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

 static struct PyModuleDef module = {
     PyModuleDef_HEAD_INIT,
     "oslo",
     "Implements the Oslo Model.",
     -1,
     methods
 };


 PyMODINIT_FUNC PyInit_oslo(void)
 {
     import_array();
     return PyModule_Create(&module);
 }
