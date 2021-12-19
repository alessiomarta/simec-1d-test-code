/******************************************************
*	
*	GENERIC LAYER CLASS HEADER
*
******************************************************/

/*! 
 *  \brief     Generic layer class.
 *  \details   The class layer is an abstract class to be used to realize different kind of layers.
 */

#ifndef LAYER_H 
#define LAYER_H 

//STD libs
#include<vector>

//OpenMP libs
#include<omp.h>

//Eigen
#include"Eigen/Dense"

//NeuranNet headers
#include"activation_functions.h"

/******************************************************
*	Generic layer class
******************************************************/
class layer
{
	public:
	//Ctors
	/**Default constructor. */
	layer(){	};
	/**Initializes a generic layer given a matrix of weights and a vector of biases. */
	layer(const std::vector<std::vector<double>> &weights_,const std::vector<double> &biases_);
	/**Initializes a generic layer given a matrix of weights and a vector of biases. */
	layer(const std::vector<std::vector<double>> &weights_);
	/**Initializes a generic layer given a matrix of weights and a vector of biases. */
	layer(const Eigen::MatrixXd &weights_);
	/**Initializes a generic layer given a matrix of weights and a vector of biases. */
	layer(const Eigen::MatrixXd &weights_,const std::vector<double> &biases_);
	/**Initializes a generic layer given a matrix of weights and a vector of biases. */
	layer(const Eigen::MatrixXd &weights_,const Eigen::VectorXd &biases_);
	/**Initializes a generic layer given a matrix of weights and a vector of biases. */
	layer(const Eigen::MatrixXd &weights_,const Eigen::VectorXd &biases_, int in_dim, int out_dim);

	//Dtor
	/**Default deconstructor. */
	virtual ~layer(){	};
	
	//Compute the output of the layer given an input
	/**Computes the output of the layer given an Eigen::VectorXd as input. */
	virtual Eigen::VectorXd predict(const Eigen::VectorXd & input) {return input;};
	/**Computes the outputs of the layer given a vector<Eigen::VectorXd> as input. */
	virtual std::vector<Eigen::VectorXd> predict_batch(const std::vector<Eigen::VectorXd> & input) {return input;};
	
	//Functions to get biases and weights from the network
	/** Returns the biases of the layer. */
	const Eigen::VectorXd get_biases();
	/** Returns the weights of the layer. */
	const Eigen::MatrixXd get_weights();
	/** Returns the weights and the biases of the layers in a matrix whose last column is the vector of the biases. */
	const Eigen::MatrixXd get_weights_biases_as_mat();
	/** Returns the weights and the biases of the layers as a column vector. The order is the following: Weights of the first node,
	* bias of the first node, weights of the first node, bias of the second node, etc.  */
	const Eigen::MatrixXd get_weights_biases_as_vec_col_maj();
	/** Returns the weights and the biases of the layers as a row vector. The order is the following: Weights of the first node,
	* bias of the first node, weights of the first node, bias of the second node, etc.  */
	const Eigen::MatrixXd get_weights_biases_as_vec_row_maj();
	/** Returns the (i-th,j-th) entry of the matrix of weights. */
	double get_weight(int i, int j);
	/** Returns the bias of the i-th node. */
	double get_bias(int i);
	
	//Set biases and weights (supposing they are alread
	/** Sets the biases of the layer. */
	void set_biases(const Eigen::VectorXd & biases_);
	/** Sets the biases of the layer. */
	void set_biases(const std::vector<double> & biases_);
	/** Sets the weights of the layer. */
	void set_weights(const Eigen::MatrixXd & weights_);
	/** Sets the (i-th,j-th) weight of the layer. */
	void set_weight(int i, int j, double weight_);
	/** Sets the bias of the i-th node. */
	void set_bias(int i,double bias_);
	/** Sets the weights and the biases of the layer, passing a matrix whose last column is the vector of the biases.  */
	void set_weights_biases_compact(Eigen::MatrixXd &source);
	/** Sets the weights and the biases of the layer. The entries of the input (1 x N) matrix is the following: Weights of the first node,
	* bias of the first node, weights of the first node, bias of the second node, etc.  */
	void set_weights_biases(Eigen::MatrixXd &source);
	/** Sets the weights and the biases of the layer. The entries of the input (1 x N) matrix is the following: Weights of the first node,
	* bias of the first node, weights of the first node, bias of the second node, etc.  */
	void set_weights_biases_row_maj(Eigen::MatrixXd &source);

	//Get info about the layer
	/** Returns the number of nodes of the layer. */
	int get_num_nodes();
	/** Returns the dimension of the input space of the layer. */
	int get_input_size();
	/** Returns the number of rows of the weights matrix. */
	int get_weights_rows();
	/** Returns the number of columns of the weights matrix. */
	int get_weights_cols();
	/** Returns the type of the layer. */
	std::string get_type();
		
	//Compute Jacobians w.r.t. weights/biases or inputs
	/**Computes the Jacobian matrix with respect to the input variables. The Jacobian is computer in reduced form. */
	virtual Eigen::MatrixXd compute_partial_derivatives_wrt_inputs(Eigen::VectorXd &input) = 0;
	/**Computes the Jacobian matrix with respect to weights and biases. */
	virtual Eigen::MatrixXd compute_partial_derivatives_wrt_weights_biases(Eigen::VectorXd &input) = 0;

	//Utils
	/**Transposes the weights matrix of the layer. */
	void transpose_weights();
	
	//Set the parameters to approximate x -> exp(-x) and x-> ln(1+exp(x)) for large values of the argument
	//See the description of the variables exp_max_cut and exp_zero_approx
	/** Set the exp_max_cut parameter to approximate \f$ f(x) = e^{-x} \f$ and \f$ f(x) = ln(1+e^{x}) \f$ for large values of the argument \f$ x \f$.
	* - For \f$ x \f$ above exp_max_cut approximates \f$ f(x) = e^{-x} \f$ with exp_zero_approx.
	* - For \f$ x \f$ above exp_max_cut approximates \f$ f(x) =  ln(1+e^{x}) \f$ with \f$ x \f$.
	*/
	void set_exp_max_cut(double exp_max_cut_);
	/** Set the exp_zero_approx parameter to approximate \f$ f(x) = e^{-x} \f$ and \f$ f(x) = ln(1+e^{x}) \f$ for large values of the argument \f$ x \f$.
	* - For \f$ x \f$ above exp_max_cut approximates \f$ f(x) = e^{-x} \f$ with exp_zero_approx.
	* - For \f$ x \f$ above exp_max_cut approximates \f$ f(x) =  ln(1+e^{x}) \f$ with \f$ x \f$.
	*/
	void set_exp_zero_approx(double exp_zero_approx_);
	
	//Get the parameters to approximate x -> exp(-x) and x->ln(1+exp(x)) for large arguments
	/** Returns the parameter exp_max_cut, controlling the approximations of \f$ f(x) = e^{-x} \f$ and \f$ f(x) = ln(1+e^{x}) \f$ for large values of the argument \f$ x \f$. */
	double get_exp_max_cut();
	/** Returns the parameter exp_zero_approx, approximating \f$ f(x) = e^{-x} \f$ for large values of the argument \f$ x \f$. */
	double get_exp_zero_approx();
	 
	protected:
	//Biases and weights of the neural network
	Eigen::VectorXd biases;
	Eigen::MatrixXd weights;
	//Dimensions of the input/output spaces and of the weights matrix
	int weights_rows = 0;
	int input_dim = 0;
	int output_dim = 0;	
	int weights_cols = 0;
	//Type of layer
	std::string type; 
	//Parameters to approximate x -> exp(-x) and x-> ln(1+exp(x)) large arguments
	double exp_max_cut = 50.;
	double exp_zero_approx = 1.92874984e-22;

};

#endif /* layer_H */

