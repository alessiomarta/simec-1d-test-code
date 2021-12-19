/******************************************************
*	
*		NEURAL NETWORK CLASS
*
******************************************************/

/*! 
 *  \brief     The neural network class.
 */


#ifndef NEURAL_NETWORK_H 
#define NEURAL_NETWORK_H 


//STD libs
#include<math.h>
#include<vector>
#include<fstream>
#include<iostream>
#include<string>

//OpenMP libs
#include<omp.h>

//Eigen libs
#include"Eigen/Dense"

//Neural network libs
#include"layers.h"
#include"layer.h"

using namespace std;

/** Neural network class.*/
class neural_network
{
	public:
	//ctor
	/**Default constructor. */
	neural_network(){num_layers = 0;};
	/**Builds a neural network given a vectors of pointers to layers. */
	neural_network(const std::vector<layer*> &ls);
	
	//dtor
	/**Default deconstructor */
	~neural_network(){ for (int i = 0; i < num_layers; i++){ delete layers[i];}};
	
	//Info about the neural network
	
	/**Returns the number of layers of the neural network. */
	int get_num_layers();
	/**Prints info about the neural network. */
	void print_network_info();
	/**Prints detailed info about the neural network. */
	
	//Prediction functions
	
	void print_network_detailed_info();
	//Prediction functions (serial version). 
	/**Computes the output of the neural network for a given input. */
	Eigen::VectorXd predict(const Eigen::VectorXd &input);
	/**Computes the output of the subnetwork up to a specified layer for a given input. */
	Eigen::VectorXd predict_to_layer(int tolayer,const Eigen::VectorXd &input);
	//Prediction functions - Parallel version
	/**Computes the outputs of the neural network for a given vector of inputs. */
	std::vector<Eigen::VectorXd> predict(const std::vector<Eigen::VectorXd> &input);
	/**Computes the outputs of the subnetwork up to a specified layer for a given vector of inputs. */
	std::vector<Eigen::VectorXd> predict_to_layer(int tolayer,const std::vector<Eigen::VectorXd> &input);
	
	//Save/read the network
	
	/**Reads network from file*/
	void network_read(const std::string &filename);
	/**Saves network to file*/
	void network_save(const std::string &filename);
	
	//Builds the network
	/**
	* Append a layer to the neural network. 
	* Arguments:
	* - wrows: Number of rows of the weights matrix.
	* - wcols: Number of columns of the weights matrix.
	* - in_dim: dimension of the input space of the layer.
	* - out_dim: dimension of the output space of the layer.
	* - type: type of layer. 
	* .
	* Note: For feedforward layers wrows = in_dim and wcols = out_dim.
	*/
	void add_layer_all_parameters(int wrows, int wcols, int in_dim, int out_dim, std::string type);
	/** 
	* Builds the input layer of a fully-connected feedforward layer to the neural network. 
	* Arguments:
	* - in_dim: dimension of the input space of the layer.
	* - out_dim: dimension of the output space of the layer.
	* - type: type of layer.
	* .
	*/
	void add_fc_input_layer(int in_dim, int out_dim, std::string type);
	/** 
	* Adds an hidden layer of a fully-connected feedforward layer to the neural network. 
	* Arguments:
	* - in_dim: dimension of the input space of the layer.
	* - out_dim: dimension of the output space of the layer.
	* - type: type of layer.
	* .
	* Note: The last hidden layer built invoking this method is the final layer of the network.
	*/
	void add_fc_hidden_layer(int out_dim, std::string type);
	
	//Set and get biases and weights. To be used with the python wrapper
	/** Sets the weights of a given layer given a matrix in input. */
	void set_layer_weights(int num_layer, Eigen::MatrixXd& weights_);
	/** Sets the biases of a given layer given a vector in input. */
	void set_layer_biases(int num_layer, Eigen::VectorXd& biases_);
	/** Returns a copy of the weights matrix of a given layer. */
	Eigen::MatrixXd get_layer_weights(int num_layer);
	/** Returns a copy of the biases vector of a given layer. */
	Eigen::VectorXd get_layer_biases(int num_layer);
	
	//Loss functions
	double loss_mse(const std::vector<Eigen::VectorXd> &data, const std::vector<Eigen::VectorXd> &features);
	double accuracy(const std::vector<Eigen::VectorXd> &data, const std::vector<Eigen::VectorXd> &features);
	
	//SiMEC 1D algos
	
	/**
	* Basic version of the SiMEC_1D algorithm, bulding the equaivalence class of a point. \n
	* Arguments:
	* - output_file: Name of the file in which the outputs of the algorithm are saved.
	* - steps: Number of interations of the algorithm.
	* - delta: Integration step of the algorithm.
	* - invert_build: Allows to follow a null curve in both directions.
	* - ipoint: The starting point for which SiMEC_1D builds the equivalence class.
	*/
	void SiMEC_1D(std::string output_file, int steps, double delta, bool invert_build, Eigen::VectorXd ipoint);
	/**
	* A modification of the SiMEC_1D algorithm. When it hits the boundary it projects inside the hypercube \f$(a_1\cdots,a_n) \times (b_1,\cdots,b_n) \f$. \n
	* Arguments:
	* - output_file: Name of the file in which the outputs of the algorithm are saved.
	* - steps: Number of interations of the algorithm.
	* - delta: Integration step of the algorithm.
	* - invert_build: Allows to follow a null curve in both directions.
	* - ipoint: The starting point for which SiMEC_1D builds the equivalence class.
	* - H_inf: The vector \f$(a_1\cdots,a_n)\f$.
	* - H_max: The vector \f$(b_1,\cdots,b_n) \f$
	*/
	void SiMEC_1D_norm_proj(std::string output_file, int steps, double delta, bool invert_build, Eigen::VectorXd ipoint, Eigen::VectorXd &H_inf, Eigen::VectorXd &H_sup);
	/**
	* A modification of the SiMEC_1D algorithm. Hitting the boundary of the hypercube \f$(a_1\cdots,a_n) \times (b_1,\cdots,b_n) \f$ halts the algorithm. \n
	* Arguments:
	* - output_file: Name of the file in which the outputs of the algorithm are saved.
	* - steps: Number of interations of the algorithm.
	* - delta: Integration step of the algorithm.
	* - invert_build: Allows to follow a null curve in both directions.
	* - ipoint: The starting point for which SiMEC_1D builds the equivalence class.
	* - H_inf: The vector \f$(a_1\cdots,a_n)\f$.
	* - H_max: The vector \f$(b_1,\cdots,b_n) \f$
	*/
	void SiMEC_1D_stop_boundary(std::string output_file, int steps, double delta, bool invert_build, Eigen::VectorXd ipoint, Eigen::VectorXd &H_inf, Eigen::VectorXd &H_sup);
	
	//SiMExp 1D algos
	/**
	*  SiMExp_1D algorithm. Produces a non-null curve of maximum lenght epsilon starting from a given point.  \n
	* Arguments:
	* - output_file: Name of the file in which the outputs of the algorithm are saved.
	* - steps: Number of interations of the algorithm.
	* - delta: Integration step of the algorithm.
	* - epsilon: Maximum distance from the starting point.
	* - invert_build: Allows to follow a null curve in both directions.
	* - max_step: The maximum number of iteration; To avoid infinite loops.
	* - ipoint: The starting point for which SiMEC_1D builds the equivalence class.
	*/
	void SiMExp_1D(std::string output_file, double delta, double epsilon, int max_steps, bool invert_build, Eigen::VectorXd & ipoint);
	/** Starting from a point, proceed along the curve which change the output of the network the most. 
	* Instead of this method, use SiMExp_1D, which avoids infinite loop by coinstruction.  
	*/
	void ClassChange_1D(std::string output_file, int steps, double delta, bool invert_build, Eigen::VectorXd ipoint);
	
	
	
	layer* get_layer(int n)
	{
		return layers[n];
	}
		
	private:
	int num_layers;
	std::vector<layer*> layers;	
};






#endif /* NEURAL_NETWORK_H */

