/******************************************************
*	
*	FEEDFORWARD LAYER WITH SOFTPLUS ACTIVATION
*			FUNCTION
*
******************************************************/

#ifndef FC_LAYER_SM_H 
#define FC_LAYER_SM_H 

//STD libs
#include<vector>

//OpenMP libs
#include<omp.h>

//Eigen libs
#include"Eigen/Dense"

//Neural network libs
#include"activation_functions.h"
#include"layer.h"

class fc_layer_sm : public layer
{
	public:
	//ctors
	fc_layer_sm( ) : layer() {	};
	fc_layer_sm(const std::vector<std::vector<double>> &weights_, const std::vector<double> &biases_);
	fc_layer_sm(const Eigen::MatrixXd &weights_, const Eigen::VectorXd &biases_);
	fc_layer_sm(const Eigen::MatrixXd &weights_, const std::vector<double> &biases_);
	//dtor
	~fc_layer_sm(){	};
	//compute the output of the layer given an input
	Eigen::VectorXd predict(const Eigen::VectorXd & input) override;
	//get info about the layer
	int get_num_nodes();
	int get_input_size();
	
	//Partial derivatives
	Eigen::MatrixXd compute_partial_derivatives_wrt_inputs(Eigen::VectorXd &input) override;
	Eigen::MatrixXd compute_partial_derivatives_wrt_weights_biases(Eigen::VectorXd &input) override;
	

};

#endif /* FC_LAYER_SM_H */

