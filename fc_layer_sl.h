/******************************************************
*	
*	FEEDFORWARD LAYER WITH SILU ACTIVATION
*			FUNCTION
*
******************************************************/

/*! 
 *  \brief     Feedforward layer with silu activation function.
 */

#ifndef FC_LAYER_SL_H 
#define FC_LAYER_SL_H 

//STD libs
#include<vector>
#include<omp.h>

//Eigen libs
#include"Eigen/Dense"

//Neural Network libs
#include"activation_functions.h"
#include"layer.h"

class fc_layer_sl : public layer
{
	public:
	
	//ctors
	/**Default constructor. */
	fc_layer_sl( ) : layer() {	};
	/**Initializes a feedforward layer with silu activation function given a matrix of weights and a vector of biases. */
	fc_layer_sl(const std::vector<std::vector<double>> &weights_,const std::vector<double> &biases_);
	/**Initializes a feedforward layer with silu activation function given a matrix of weights and a vector of biases. */
	fc_layer_sl(const Eigen::MatrixXd &weights_,const std::vector<double> &biases_);
	/**Initializes a feedforward layer with silu activation function given a matrix of weights and a vector of biases. */
	fc_layer_sl(const Eigen::MatrixXd &weights_,const Eigen::VectorXd &biases_);
	
	//dtor
	/**Default deconstructor. */
	~fc_layer_sl(){	};
	
	//Computes the output of the layer given an input
	/**Computes the output of the layer given an Eigen::VectorXd as input. */
	Eigen::VectorXd predict(const Eigen::VectorXd & input) override;
	/**Computes the outputs of the layer given a vector<Eigen::VectorXd> as input. */
	std::vector<Eigen::VectorXd> predict_batch(const std::vector<Eigen::VectorXd> & input) override;

	//Jacobians
	/**Computes the Jacobian matrix with respect to the input variables. The Jacobian is computer in reduced form. */
	Eigen::MatrixXd compute_partial_derivatives_wrt_inputs(Eigen::VectorXd &input) override;
	/**Computes the Jacobian matrix with respect to weights and biases. */
	Eigen::MatrixXd compute_partial_derivatives_wrt_weights_biases(Eigen::VectorXd &input) override;

};

#endif /* FC_LAYER_SL_H */

