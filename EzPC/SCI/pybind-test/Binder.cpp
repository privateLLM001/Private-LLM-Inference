#include <vector>
#include <complex>
#include <iostream>
#include "SCIProvider.h"

#include "../extern/pybind11/include/pybind11/pybind11.h"
#include "../extern/pybind11/include/pybind11/stl.h"
#include "../extern/pybind11/include/pybind11/complex.h"
#include "../extern/pybind11/include/pybind11/stl_bind.h"
#include "../extern/pybind11/include/pybind11/numpy.h"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<uint64_t>);
PYBIND11_MAKE_OPAQUE(std::vector<uint8_t>);

#if defined(BIT41)
    #ifdef PARTY_ALICE
        #define PACKAGE_NAME sci_provider_alice_41
    #else
        #define PACKAGE_NAME sci_provider_bob_41
    #endif
#elif defined(BIT37)
    #ifdef PARTY_ALICE
        #define PACKAGE_NAME sci_provider_alice_37
    #else
        #define PACKAGE_NAME sci_provider_bob_37
    #endif
#else
    #ifdef PARTY_ALICE
        #define PACKAGE_NAME sci_provider_alice
    #else
        #define PACKAGE_NAME sci_provider_bob
    #endif
#endif

std::vector<uint64_t> getVectorFromBuffer(py::array_t<uint64_t>& values) {
    py::buffer_info buf = values.request();
    uint64_t *ptr = (uint64_t *)buf.ptr;
    std::vector<uint64_t> vec(buf.shape[0]);
    for (auto i = 0; i < buf.shape[0]; i++)
        vec[i] = ptr[i];
    return vec;
}

py::array_t<uint64_t> getBufferFromVector(const std::vector<uint64_t>& vec) {
    py::array_t<uint64_t> values(vec.size());
    py::buffer_info buf = values.request();
    uint64_t *ptr = (uint64_t *)buf.ptr;
    for (auto i = 0; i < buf.shape[0]; i++)
        ptr[i] = vec[i];
    return values;
}

std::vector<uint8_t> getVectorFromBuffer(py::array_t<uint8_t>& values) {
    py::buffer_info buf = values.request();
    uint8_t *ptr = (uint8_t *)buf.ptr;
    std::vector<uint8_t> vec(buf.shape[0]);
    for (auto i = 0; i < buf.shape[0]; i++)
        vec[i] = ptr[i];
    return vec;
}

py::array_t<uint8_t> getBufferFromVector(const std::vector<uint8_t>& vec) {
    py::array_t<uint8_t> values(vec.size());
    py::buffer_info buf = values.request();
    uint8_t *ptr = (uint8_t *)buf.ptr;
    for (auto i = 0; i < buf.shape[0]; i++)
        ptr[i] = vec[i];
    return values;
}

PYBIND11_MODULE(PACKAGE_NAME, m) {

    py::class_<SCIProvider>(m, "SCIProvider")
        .def(py::init<int>())
        .def("startComputation", &SCIProvider::startComputation)
        .def("endComputation", &SCIProvider::endComputation)
        .def("dbits", &SCIProvider::dbits)
        .def("sqrt", [](SCIProvider& self, py::array_t<uint64_t> share, int64_t scale_in, int64_t scale_out, bool inverse) {
            auto ret = self.sqrt(getVectorFromBuffer(share), scale_in, scale_out, inverse);
            return getBufferFromVector(std::move(ret));
        })
        .def("elementwise_multiply", [](SCIProvider& self, py::array_t<uint64_t> share1, py::array_t<uint64_t> share2) {
            auto ret = self.elementwise_multiply(getVectorFromBuffer(share1), getVectorFromBuffer(share2));
            return getBufferFromVector(std::move(ret));
        })
        .def("exp", [](SCIProvider& self, py::array_t<uint64_t> share) {
            auto ret = self.exp(getVectorFromBuffer(share));
            return getBufferFromVector(std::move(ret));
        })
        .def("exp_reduce", [](SCIProvider& self, py::array_t<uint64_t> share) {
            auto ret = self.exp_reduce(getVectorFromBuffer(share));
            return getBufferFromVector(std::move(ret));
        })
        .def("softmax", [](SCIProvider& self, py::array_t<uint64_t> share, size_t dims) {
            auto ret = self.softmax(getVectorFromBuffer(share), dims);
            return getBufferFromVector(std::move(ret));
        })
        .def("div", [](SCIProvider& self, py::array_t<uint64_t> nom, py::array_t<uint64_t> den) {
            auto ret = self.div(getVectorFromBuffer(nom), getVectorFromBuffer(den));
            return getBufferFromVector(std::move(ret));
        })
        .def("tanh", [](SCIProvider& self, py::array_t<uint64_t> share) {
            auto ret = self.tanh(getVectorFromBuffer(share));
            return getBufferFromVector(std::move(ret));
        })
    ;
}