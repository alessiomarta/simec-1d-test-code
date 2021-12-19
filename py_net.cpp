#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include"neural_network.h"
#include "Eigen/Dense"
#include "Eigen/Core"

namespace py = pybind11;


PYBIND11_MODULE(py_net, m) {

    py::class_<neural_network>(m, "neural_network")
        .def(py::init())
        .def("get_num_layers", &neural_network::get_num_layers)
        .def("print_network_info", &neural_network::print_network_info)
        .def("predict", static_cast<Eigen::VectorXd (neural_network::*)(const Eigen::VectorXd &)>(&neural_network::predict), "Predict")
        .def("network_read", &neural_network::network_read)
        .def("network_save", &neural_network::network_save)
        
        //Build the layers
        .def("add_layer_all_parameters", &neural_network::add_layer_all_parameters)
        .def("add_fc_input_layer", &neural_network::add_fc_input_layer)
        .def("add_fc_hidden_layer", &neural_network::add_fc_hidden_layer)
        .def("set_layer_weights", &neural_network::set_layer_weights)
        .def("set_layer_biases", &neural_network::set_layer_biases)
        .def("get_layer_weights", &neural_network::get_layer_weights)
        .def("get_layer_biases", &neural_network::get_layer_biases)
        
        //SiMEC and SiMExp
        .def("SiMEC_1D", &neural_network::SiMEC_1D)
        .def("SiMEC_1D_norm_proj", &neural_network::SiMEC_1D_norm_proj)
        .def("SiMEC_1D_stop_boundary", &neural_network::SiMEC_1D_stop_boundary)
        .def("SiMExp_1D", &neural_network::SiMExp_1D)
        .def("ClassChange_1D", &neural_network::ClassChange_1D);
  
}

