/******************************************************
*	
*	FEEDFORWARD LAYER WITH SOFTPLUS ACTIVATION
*			FUNCTION
*
******************************************************/

/*! 
 *  \brief     Feedforward layer with softplus activation function.
 */

#ifndef FC_LAYER_SP_H 
#define FC_LAYER_SP_H 

//STD libs
#include<vector>
#include<omp.h>

//Eigen libs
#include"Eigen/Dense"

//Neural network libs
#include"activation_functions.h"
#include"layer.h"

class fc_layer_sp : public layer
{
	public:
	
	//ctors
	/**Default constructor. */
	fc_layer_sp( ) : layer() {	};
	/**Initializes a feedforward layer with softplus activation function given a matrix of weights and a vector of biases. */
	fc_layer_sp(const std::vector<std::vector<double>> &weights_, const std::vector<double> &biases_);
	/**Initializes a feedforward layer with softplus activation function given a matrix of weights and a vector of biases. */
	fc_layer_sp(const Eigen::MatrixXd &weights_, const Eigen::VectorXd &biases_);
	/**Initializes a feedforward layer with softplus activation function given a matrix of weights and a vector of biases. */
	fc_layer_sp(const Eigen::MatrixXd &weights_, const std::vector<double> &biases_);
	
	//dtor
	/**Default deconstructor. */
	~fc_layer_sp(){	};
	
	//Computes the output of the layer given an input
	/**Computes the output of the layer given an Eigen::VectorXd as input. */
	Eigen::VectorXd predict(const Eigen::VectorXd & input) override;
	/**Computes the outputs of the layer given a vector<Eigen::VectorXd> as input. */
	std::vector<Eigen::VectorXd> predict_batch(const std::vector<Eigen::VectorXd> & input) override;
	
	//Jacobians
	/**Computes the Jacobian matrix with respect to the input variables. The Jacobian is computer in reduced form. */
	Eigen::MatrixXd compute_partial_derivatives_wrt_inputs(Eigen::VectorXd &input);
	/**Computes the Jacobian matrix with respect to weights and biases. */
	Eigen::MatrixXd compute_partial_derivatives_wrt_weights_biases(Eigen::VectorXd &input);

};

#endif /* FC_layer_sp_H */

