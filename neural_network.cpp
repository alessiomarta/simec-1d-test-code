//Neural netowrk libs
#include"neural_network.h"

//Eigen libs
#include "Eigen/Dense"
#include "Eigen/Core"
#include "Eigen/Eigenvalues" 
#include "Eigen/StdVector"

//STD libs
#include <stdlib.h>   
#include <time.h>
#include <cmath>

//Utils libs
#include "vectors_utils.h"
#include "matrix_utils.h"

neural_network::neural_network(const std::vector<layer*> &ls)
{
	num_layers = ls.size();
	layers = ls;
}
//------------------------------------------------------------------------------
int neural_network::get_num_layers()
{
	return num_layers;
}
//------------------------------------------------------------------------------
void neural_network::print_network_info()
{
	std::cout << endl;
	std::cout << "*********************" << endl;
	std::cout << "*Neural network info*" << endl;
	std::cout << "*********************" << endl;
	std::cout << "Number of layers : " << num_layers << endl;
	std::cout << "----------------------------" << endl;
	if (num_layers > 0)
	{
		for (int i = 0; i < num_layers; i++)
		{
			std::cout << "Layer : " << i << endl;
			std::cout << "Input size : " << layers[i]->get_input_size() << endl;
			std::cout << "Output size : " << layers[i]->get_num_nodes() << endl;
			std::cout << "Type of layer : " << layers[i]->get_type() << endl;
			std::cout << "----------------------------" << endl;
			
		}
	}
}
//------------------------------------------------------------------------------
void neural_network::print_network_detailed_info()
{
	std::cout << "NUMBER OF LAYERS : " << num_layers << endl;
	std::cout << "*********************" << endl;
	if (num_layers > 0)
	{
		for (int i = 0; i < num_layers; i++)
		{
			std::cout << "LAYER : " << i << endl;
			std::cout << "INPUT SIZE : " << layers[i]->get_input_size() << endl;
			std::cout << "OUTPUT SIZE : " << layers[i]->get_num_nodes() << endl;
			std::cout << "BIASES" << endl;
			for (int j = 0; j < layers[i]->get_num_nodes(); j++)
			{
				std::cout << layers[i]->get_biases()[j] << " " << endl;
			}
			for (int j = 0; j < layers[i]->get_num_nodes(); j++)
			{
				std::cout << "WEIGHTS" << endl;
				for (int  k = 0; k < layers[i]->get_input_size(); k++)
				{
					double t = layers[i]->get_weights()(j,k);
					std::cout << t << " ";
				}
				std::cout << endl;
			}
			std::cout << endl;
			std::cout << "*****************" << endl;
		}
	}
}
//------------------------------------------------------------------------------	
Eigen::VectorXd neural_network::predict(const Eigen::VectorXd &input)
{
	Eigen::VectorXd res(layers[num_layers-1]->get_num_nodes());
	Eigen::VectorXd temp(input);
	for(int i = 0; i < num_layers; i++)
	{
		res = std::move(layers[i]->predict(temp));
		temp = std::move(res);
	}
	return temp;
}
//------------------------------------------------------------------------------	
Eigen::VectorXd neural_network::predict_to_layer(int tolayer,const Eigen::VectorXd &input)
{	
	Eigen::VectorXd temp(input);
	if (tolayer >= 0)
	{
		Eigen::VectorXd res(layers[tolayer]->get_num_nodes());
		for(int i = 0; i <= tolayer; i++)
		{
			res = std::move(layers[i]->predict(temp));
			temp = std::move(res);
		}
	}
	return temp;
}
//------------------------------------------------------------------------------
std::vector<Eigen::VectorXd> neural_network::predict(const std::vector<Eigen::VectorXd> &input)
{
	int num_inputs = input.size();
	//Supponising num_inputs > 0
	std::vector<Eigen::VectorXd> result(num_inputs) ;
	omp_set_num_threads(num_inputs);
	#pragma omp parallel for
	for (int i = 0; i < num_inputs; i++)
	{
		Eigen::VectorXd temp(input[0].size());
		temp = input[i];
		for(int j = 0; j < num_layers; j++)
		{
			temp = std::move(layers[j]->predict(temp));
		}
		result[i] = std::move(temp);
		
	}
	return result;
}
//------------------------------------------------------------------------------	
std::vector<Eigen::VectorXd> neural_network::predict_to_layer(int tolayer,const std::vector<Eigen::VectorXd> &input)
{	
	int num_inputs = input.size();
	std::vector<Eigen::VectorXd> result(input);
	if (tolayer >= 0)
	{
		omp_set_num_threads(num_inputs);
		#pragma omp parallel for
		for (int i = 0; i < num_inputs; i++)
		{
			Eigen::VectorXd temp(input[0].size());
			temp = input[i];
			for(int j = 0; j <= tolayer; j++)
			{
				temp = std::move(layers[j]->predict(temp));
			}
			result[i] = std::move(temp);
		}
	}
	return result;
}
//------------------------------------------------------------------------------		
void neural_network::network_read(const string &filename)
{
	std::vector<string> type_of_layer;
	string line;
	std::ifstream file(filename);
	std::vector<string> lines;
	//Saving the lines of the file in a std::vector<string>
	if (file.is_open())
	{
		int nline = 0;
		while (getline(file,line))
		{
			lines.push_back(line);
			nline++;
		}
		file.close();
	}
	else std::cout << "File not found";
	
	std::vector<std::vector<double>> vec_biases;
	std::vector<Eigen::MatrixXd> mats_weights;
	std::vector<int> d_in;
	std::vector<int> d_out;
	
	//Extracting number of layers
	line = lines[0];
	std::stringstream sstr0(line);
	sstr0 >> num_layers;
	int nlayers = num_layers;
	type_of_layer.resize(num_layers);
	d_in.resize(num_layers);
	d_out.resize(num_layers);	
	
	//Reading layers data
	double valf;
	std::cout << "NUMBER OF LAYERS :: " << nlayers << endl;
	for (int i = 0; i < nlayers; i++)
	{
		std::cout << "Analizying Layer : " << i << endl; 
		int nrows;
		int ncols;
		//Reading number of nodes and input dimension
		std::stringstream sstr1(lines[1+3*i]);
		sstr1 >> nrows;
		sstr1 >> ncols;
		sstr1 >> d_in[i];
		sstr1 >> d_out[i];
		sstr1 >> type_of_layer[i];
		
		std::cout << "TYPE: " << type_of_layer[i] << endl;
		
		//Reading the weights
		std::stringstream sstr2(lines[2+3*i]);
		std::vector<double> temp;
		while(sstr2 >> valf)
		{
		    temp.push_back(valf);
		}
		int len = temp.size();
		
		//Creating the matrix of the weights
		Eigen::MatrixXd mat(nrows,ncols);
		//#pragma omp parallel for
		for (int i = 0; i < len; i++)
		{
    		mat(i/ncols,i%ncols) = temp[i];
		}
		mats_weights.push_back(mat);
		
		//Reading the biases
		//std::cout << "READING THE BIASES : " << endl;
		std::stringstream sstr3(lines[3+3*i]);
		temp.clear();
		while(sstr3 >> valf)
		{
   		 	temp.push_back(valf);
		}
		vec_biases.push_back(temp);
		//std::cout << "LENGTH OF BIASES: " << temp.size() << endl;
 	}
	
	layers.resize(num_layers);
	//Building the layers
	for (int i = 0; i < nlayers; i++)
	{
		if (type_of_layer[i]=="FC_LAYER_SG")
		{
			layers[i] = new fc_layer_sg(mats_weights[i],vec_biases[i]);
		}
		if (type_of_layer[i]=="FC_LAYER_SM")
		{
			layers[i] = new fc_layer_sm(mats_weights[i],vec_biases[i]); 
		}
		if (type_of_layer[i]=="FC_LAYER_SP")
		{
			layers[i] = new fc_layer_sp(mats_weights[i],vec_biases[i]); 
		}
		if (type_of_layer[i]=="FC_LAYER_SL")
		{
			layers[i] = new fc_layer_sl(mats_weights[i],vec_biases[i]); 
		}

	}
}
//------------------------------------------------------------------------------
void neural_network::network_save(const string &filename)
{
	std::ofstream out_file;
	out_file.open(filename);
	
	if (out_file.is_open())
	{	
		out_file << num_layers << endl;
		string type;
		
		for (int i = 0; i < num_layers; i++)
		{
			//Number of rows and columns of the weights matrix, dimensions of input and output and type of layer  
			type = layers[i]->get_type();
			out_file << layers[i]->get_weights_rows() << "\t" << layers[i]->get_weights_cols() << "\t" << layers[i]->get_num_nodes() << "\t" << layers[i]->get_input_size() << "\t" << type << endl;
			//Write the weights
			int nrows = layers[i]->get_weights_rows();
			int ncols = layers[i]->get_weights_cols();
			for (int j = 0; j < nrows-1; j++)
			{
				for (int k = 0; k < ncols; k++)
				{
					out_file << layers[i]->get_weight(j,k) << "\t";
				}
			}
			for (int k = 0; k < ncols-1; k++)
			{
				out_file << layers[i]->get_weight(nrows-1,k) << "\t";
			}
			out_file << layers[i]->get_weight(nrows-1,ncols-1) << endl;
			//Write the biases
			for (int k = 0; k < nrows-1; k++)
			{
				out_file << layers[i]->get_bias(k) << "\t";
			}
			out_file << layers[i]->get_bias(nrows-1) << endl;
			
		}
	out_file.close();
	}
	else
	{
		std::cout << "Cannot save the network" << endl;
	}
}
//------------------------------------------------------------------------------	
void neural_network::add_layer_all_parameters(int wrows, int wcols, int in_dim, int out_dim, string type)
{
	srand((unsigned int) time(0));
	if (type == "FC_LAYER_SG")
	{
		//Uniform initialization in (-1,1)
		num_layers++;
		layers.resize(num_layers);
		Eigen::MatrixXd mat_weights = std::move(Eigen::MatrixXd::Random(wrows,wcols));
		Eigen::VectorXd vec_biases = std::move(Eigen::VectorXd::Zero(wrows));
		layers[num_layers-1] = new fc_layer_sg(mat_weights,vec_biases);
		std::cout << type << " layer with (" << wrows << " x " << wcols << ") weights matrix, input dimension " << wcols << " and output dimension " << wrows << " created" << endl;
	}
	else if (type == "FC_LAYER_SM")
	{
		num_layers++;
		layers.resize(num_layers);
		Eigen::MatrixXd mat_weights = std::move(Eigen::MatrixXd::Random(wrows,wcols));
		Eigen::VectorXd vec_biases = std::move(Eigen::VectorXd::Zero(wrows));
		layers[num_layers-1] = new fc_layer_sm(mat_weights,vec_biases); 
		std::cout << type << " layer with (" << wrows << " x " << wcols << ") weights matrix, input dimension " << wcols << " and output dimension " << wrows << " created" << endl;
	}
	else if (type == "FC_LAYER_SP")
	{
		num_layers++;
		layers.resize(num_layers);
		Eigen::MatrixXd mat_weights = std::move(Eigen::MatrixXd::Random(wrows,wcols));
		Eigen::VectorXd vec_biases = std::move(Eigen::VectorXd::Zero(wrows));
		layers[num_layers-1] = new fc_layer_sp(mat_weights,vec_biases); 
		std::cout << type << " layer with (" << wrows << " x " << wcols << ") weights matrix, input dimension " << wcols << " and output dimension " << wrows << " created" << endl;
	}
	else if (type == "FC_LAYER_SL")
	{
		num_layers++;
		layers.resize(num_layers);
		Eigen::MatrixXd mat_weights = Eigen::MatrixXd::Random(wrows,wcols);
		Eigen::VectorXd vec_biases = Eigen::VectorXd::Zero(wrows);
		layers[num_layers-1] = new fc_layer_sl(mat_weights,vec_biases); 
		std::cout << type << " layer with (" << wrows << " x " << wcols << ") weights matrix, input dimension " << wcols << " and output dimension " << wrows << " created" << endl;
	}
	else
	{
		std::cout << "Type of layer not known" << endl;
	}
}
//------------------------------------------------------------------------------
void neural_network::add_fc_input_layer(int in_dim, int out_dim, string type)
{
	add_layer_all_parameters(out_dim, in_dim, in_dim, out_dim, type);
}
//------------------------------------------------------------------------------
void neural_network::add_fc_hidden_layer(int out_dim, string type)
{
	int in_dim = layers[num_layers-1]->get_num_nodes();
	add_layer_all_parameters(out_dim, in_dim, in_dim, out_dim, type);	
}
//------------------------------------------------------------------------------
void neural_network::set_layer_weights(int num_layer, Eigen::MatrixXd& weights_)
{
	layers[num_layer]->set_weights(weights_);
}
//------------------------------------------------------------------------------
void neural_network::set_layer_biases(int num_layer, Eigen::VectorXd& biases_)
{
	layers[num_layer]->set_biases(biases_);
}
//------------------------------------------------------------------------------
Eigen::MatrixXd neural_network::get_layer_weights(int num_layer)
{
	return layers[num_layer]->get_weights();
}
//------------------------------------------------------------------------------
Eigen::VectorXd neural_network::get_layer_biases(int num_layer)
{
	return layers[num_layer]->get_biases();
}
//------------------------------------------------------------------------------	
void neural_network::SiMEC_1D(string output_file, int steps, double delta, bool invert_build, Eigen::VectorXd ipoint)
{

	//Dimensions of input and output manifolds of the neural network
	const int dim_input = layers[0]->get_input_size();
	const int dim_output = layers[num_layers-1]->get_num_nodes();

	//Initialize previous eigenvector to keep track of the direction. 
	//Before the first iteration we set it to the zero vector
	Eigen::VectorXd old_eigenv = Eigen::VectorXd::Zero(dim_input);
	
	//Temp variables needed for the computations
	Eigen::MatrixXd temp_metric;
	Eigen::MatrixXd jacobian;
	Eigen::MatrixXd jacobian_t;
	Eigen::VectorXd fpoint;
	double energy = 0.;
	
	//Open the file in which we save the results of the algorithm
	std::ofstream file;
	file.open(output_file);
	//Size of the VectorXd to write
	const int len = ipoint.size();
	//Separating char in file
	char separating_char = '\t';
	
	//The algorithm starts
	for (int i = 0; i < steps; i++)
	{
		std::cout << "Step : " << i << " / " << steps << endl;
		
		//Initialize final Euclidean metric
		Eigen::MatrixXd fmetric = Eigen::MatrixXd::Identity(dim_output,dim_output);
		
		//Compute the pullback of the final Euclidean metric
		for (int j = num_layers-1; j > -1; j--)
		{
			fpoint = std::move(predict_to_layer(j-1,ipoint));
			jacobian = std::move(layers[j]->compute_partial_derivatives_wrt_inputs(fpoint));
			jacobian_t = std::move(jacobian.transpose());
			Eigen::MatrixXd new_metric = std::move(jacobian_t*fmetric*jacobian);
			fmetric = std::move(new_metric);
		}
		
		//Find eigenvalues and eigenvectots
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(fmetric);
		Eigen::MatrixXd eigen_vec = std::move(es.eigenvectors());
		
		//The first eigenvector is the lowest one
		Eigen::VectorXd null_eigen = eigen_vec.col(0);
	
		//Make sure to always proceed in the same direction
		if (old_eigenv.dot(eigen_vec.col(0)) < 0.)
		{
			null_eigen = (-1.)*null_eigen;
		}
		
		//Fix the direction as specified by the user
		if(invert_build == false)
		{
			ipoint = ipoint+delta*null_eigen;
		}
		else
		{
			ipoint = ipoint-delta*null_eigen;
		}

		//Compute the new energy
		energy += delta*(null_eigen.dot(fmetric*null_eigen));
		
		//Print energy and prediction
		std::cout << "Energy: " << energy << endl;
		std::cout << "Prediction: "<< predict(ipoint) << endl;
		std::cout << "POINT:" << endl;
		std::cout << ipoint << endl;
		std::cout << "****" << endl;
		//Save the eigenvector of this iteration to compare with the next one,
		//in order to proceed in the same sense.
		old_eigenv = null_eigen;
		
		//Save the result in the desired file (in csv format)
		for (int i = 0; i < len-1; i++)
		{
			file << ipoint(i);
			file << separating_char;
		}
		file << ipoint(len-1);
		file << "\n";
	}
	
}
//------------------------------------------------------------------------------
void neural_network::SiMEC_1D_norm_proj(string output_file, int steps, double delta, bool invert_build, Eigen::VectorXd ipoint, Eigen::VectorXd &H_inf, Eigen::VectorXd &H_sup)
{

	//Dimensions of input and output manifolds of the neural network
	const int dim_input = layers[0]->get_input_size();
	const int dim_output = layers[num_layers-1]->get_num_nodes();

	//Initialize previous eigenvector to keep track of the direction. 
	//Before the first iteration we set it to the zero vector
	Eigen::VectorXd old_eigenv = Eigen::VectorXd::Zero(dim_input);
	
	//Temp variables needed for the computations
	Eigen::MatrixXd temp_metric;
	Eigen::MatrixXd jacobian;
	Eigen::MatrixXd jacobian_t;
	Eigen::VectorXd fpoint;
	double energy = 0.;
	
	//Open the file in which we save the results of the algorithm
	std::ofstream file;
	file.open(output_file);
	//Size of the VectorXd to write
	const int len = ipoint.size();
	//Separating char in file
	char separating_char = '\t';
	
	//The algorithm starts
	for (int i = 0; i < steps; i++)
	{
		std::cout << "Step : " << i << " / " << steps << endl;
		
		//Initialize final Euclidean metric
		Eigen::MatrixXd fmetric = Eigen::MatrixXd::Identity(dim_output,dim_output);
		
		//Compute the pullback of the final Euclidean metric
		for (int j = num_layers-1; j > -1; j--)
		{
			fpoint = std::move(predict_to_layer(j-1,ipoint));
			jacobian = std::move(layers[j]->compute_partial_derivatives_wrt_inputs(fpoint));
			jacobian_t = std::move(jacobian.transpose());
			Eigen::MatrixXd new_metric = std::move(jacobian_t*fmetric*jacobian);
			fmetric = std::move(new_metric);
		}
		
		//Find eigenvalues and eigenvectots
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(fmetric);
		Eigen::MatrixXd eigen_vec = std::move(es.eigenvectors());
		
		//The first eigenvector is the lowest one
		Eigen::VectorXd null_eigen = eigen_vec.col(0);
	
		//Make sure to always proceed in the same direction
		if (old_eigenv.dot(eigen_vec.col(0)) < 0.)
		{
			null_eigen = (-1.)*null_eigen;
		}
		
		//Fix the direction as specified by the user
		if(invert_build == false)
		{
			ipoint = ipoint+delta*null_eigen;
		}
		else
		{
			ipoint = ipoint-delta*null_eigen;
		}

		//Project in (-1,1)^n
		for (int i = 0; i < dim_input; i++)
		{
			if (ipoint(i) > H_sup(i)) ipoint(i) = H_sup(i);
			if (ipoint(i) < H_inf(i)) ipoint(i) = H_inf(i);
		}

		//Compute the new energy
		energy += delta*(null_eigen.dot(fmetric*null_eigen));
		
		//Print energy and prediction
		std::cout << "Energy: " << energy << endl;
		std::cout << "Prediction: "<< predict(ipoint) << endl;
		std::cout << "POINT:" << endl;
		std::cout << ipoint << endl;
		std::cout << "****" << endl;
		//Save the eigenvector of this iteration to compare with the next one,
		//in order to proceed in the same sense.
		old_eigenv = null_eigen;
		
		//Save the result in the desired file (in csv format)
		for (int i = 0; i < len-1; i++)
		{
			file << ipoint(i);
			file << separating_char;
		}
		file << ipoint(len-1);
		file << "\n";
	}
	
}
//------------------------------------------------------------------------------	
void neural_network::SiMEC_1D_stop_boundary(std::string output_file, int steps, double delta, bool invert_build, Eigen::VectorXd ipoint, Eigen::VectorXd &H_inf, Eigen::VectorXd &H_sup)
{

	//Dimensions of input and output manifolds of the neural network
	const int dim_input = layers[0]->get_input_size();
	const int dim_output = layers[num_layers-1]->get_num_nodes();

	//Initialize previous eigenvector to keep track of the direction. 
	//Before the first iteration we set it to the zero vector
	Eigen::VectorXd old_eigenv = Eigen::VectorXd::Zero(dim_input);
	
	//Temp variables needed for the computations
	Eigen::MatrixXd temp_metric;
	Eigen::MatrixXd jacobian;
	Eigen::MatrixXd jacobian_t;
	Eigen::VectorXd fpoint;
	double energy = 0.;
	
	//Open (in append mode) the file in which we save the results of the algorithm
	std::ofstream file;
	file.open(output_file,std::ios_base::app | std::ios_base::out);
	//Size of the VectorXd to write
	const int len = ipoint.size();
	//Separating char in file
	char separating_char = '\t';
	//flag var telling when the boundary is hit
	bool halt = false;
	
	//The algorithm starts
	for (int i = 0; i < steps; i++)
	{
		//std::cout << "Step : " << i << " / " << steps << endl;
		
		//Initialize final Euclidean metric
		Eigen::MatrixXd fmetric = Eigen::MatrixXd::Identity(dim_output,dim_output);
		
		//Compute the pullback of the final Euclidean metric
		for (int j = num_layers-1; j > -1; j--)
		{
			fpoint = std::move(predict_to_layer(j-1,ipoint));
			jacobian = std::move(layers[j]->compute_partial_derivatives_wrt_inputs(fpoint));
			jacobian_t = std::move(jacobian.transpose());
			Eigen::MatrixXd new_metric = std::move(jacobian_t*fmetric*jacobian);
			fmetric = std::move(new_metric);
		}
		
		//Find eigenvalues and eigenvectots
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(fmetric);
		Eigen::MatrixXd eigen_vec = std::move(es.eigenvectors());
		
		//The first eigenvector is the lowest one
		Eigen::VectorXd null_eigen = eigen_vec.col(0);
	
		//Make sure to always proceed in the same direction
		if (old_eigenv.dot(eigen_vec.col(0)) < 0.)
		{
			null_eigen = (-1.)*null_eigen;
		}
		
		//Fix the direction as specified by the user
		if(invert_build == false)
		{
			ipoint = ipoint+delta*null_eigen;
		}
		else
		{
			ipoint = ipoint-delta*null_eigen;
		}

		//Stop at the boundary
		
		for (int i = 0; i < dim_input; i++)
		{
			if (ipoint(i) > H_sup(i))
			{
				ipoint(i) = H_sup(i);
				halt = true;
			}
			if (ipoint(i) < H_inf(i))
			{
				ipoint(i) = H_inf(i);
				halt = true;	
			}
		}

		//Compute the new energy
		energy += delta*(null_eigen.dot(fmetric*null_eigen));
		
		/*
		//Print energy and prediction
		std::cout << "Energy: " << energy << endl;
		std::cout << "Prediction: "<< predict(ipoint) << endl;
		std::cout << "POINT:" << endl;
		std::cout << ipoint << endl;
		std::cout << "****" << endl;
		*/
		
		//Save the eigenvector of this iteration to compare with the next one,
		//in order to proceed in the same sense.
		old_eigenv = null_eigen;
		
		//Save the result in the desired file (in csv format)
		for (int i = 0; i < len-1; i++)
		{
			file << ipoint(i);
			file << separating_char;
		}
		file << ipoint(len-1);
		file << "\n";
		
		//If the boundary is hit, stop
		if (halt == true)
		{
			break;
		}
	}
	file.close();
	
}
//------------------------------------------------------------------------------
void neural_network::SiMExp_1D(std::string output_file, double delta, double epsilon, int max_steps, bool invert_build, Eigen::VectorXd & ipoint)
{

	//Dimensions of input and output manifolds of the neural network
	const int dim_input = layers[0]->get_input_size();
	const int dim_output = layers[num_layers-1]->get_num_nodes();

	//Initialize previous eigenvector to keep track of the direction. 
	//Before the first iteration we set it to the zero vector
	Eigen::VectorXd old_eigenv = Eigen::VectorXd::Zero(dim_input);
	
	//Temp variables needed for the computations
	Eigen::MatrixXd temp_metric;
	Eigen::MatrixXd jacobian;
	Eigen::MatrixXd jacobian_t;
	Eigen::VectorXd fpoint;
	
	
	//Open the file in which we save the results of the algorithm
	std::ofstream file;
	file.open(output_file,std::ios_base::app | std::ios_base::out);
	//Size of the VectorXd to write
	const int len = ipoint.size();
	//Separating char in file
	char separating_char = '\t';
	//Starting point output
	double starting_ouput = predict(ipoint)(0);
	//Count the number of iteration to avoid infinite loop
	int step = 0;	
	
	do
	{
		Eigen::MatrixXd fmetric = Eigen::MatrixXd::Identity(dim_output,dim_output);
		
		//Compute the pullback of the final Euclidean metric
		for (int j = num_layers-1; j > -1; j--)
		{
			fpoint = std::move(predict_to_layer(j-1,ipoint));
			jacobian = std::move(layers[j]->compute_partial_derivatives_wrt_inputs(fpoint));
			jacobian_t = std::move(jacobian.transpose());
			Eigen::MatrixXd new_metric = std::move(jacobian_t*fmetric*jacobian);
			fmetric = std::move(new_metric);
		}
		
		//Find eigenvalues and eigenvectots
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(fmetric);
		Eigen::MatrixXd eigen_vec = std::move(es.eigenvectors());
		
		//The last eigenvector is the one along which the e.c. change more
		Eigen::VectorXd max_eigen = eigen_vec.col(dim_input-1);

		//Make sure to always proceed in the same direction
		if (old_eigenv.dot(eigen_vec.col(dim_input-1)) < 0.)
		{
			max_eigen = (-1.)*max_eigen;
		}
		
		//Fix the direction as specified by the user
		if(invert_build == false)
		{
			ipoint = ipoint+delta*max_eigen;
		}
		else
		{
			ipoint = ipoint-delta*max_eigen;
		}
		
		step++;
		
		std::cout << "Exp Prediction: " << predict(ipoint) << endl;
		std::cout << "Starting prediction: " << starting_ouput << endl;
		//Save the result in the desired file (in csv format)
		for (int i = 0; i < len-1; i++)
		{
			file << ipoint(i);
			file << separating_char;
		}
		file << ipoint(len-1);
		file << "\n";
	}
	while (abs(starting_ouput-predict(ipoint)(0)) < epsilon && step < max_steps);
	
	file.close();
}
//------------------------------------------------------------------------------
void neural_network::ClassChange_1D(string output_file, int steps, double delta, bool invert_build, Eigen::VectorXd ipoint)
{

	//Dimensions of input and output manifolds of the neural network
	const int dim_input = layers[0]->get_input_size();
	const int dim_output = layers[num_layers-1]->get_num_nodes();

	//Initialize previous eigenvector to keep track of the direction. 
	//Before the first iteration we set it to the zero vector
	Eigen::VectorXd old_eigenv = Eigen::VectorXd::Zero(dim_input);
	
	//Temp variables needed for the computations
	Eigen::MatrixXd temp_metric;
	Eigen::MatrixXd jacobian;
	Eigen::MatrixXd jacobian_t;
	Eigen::VectorXd fpoint;
	
	
	//Open the file in which we save the results of the algorithm
	std::ofstream file;
	file.open(output_file);
	//Size of the VectorXd to write
	const int len = ipoint.size();
	//Separating char in file
	char separating_char = '\t';
	double energy = 0.;
	
	/* while (prediction(0)-epsilon > 0.)
	 Esegui algoritmo
	*/ 
	for (int i = 0; i < steps; i++)
	{
		std::cout << "Step : " << i << " / " << steps << endl;
			
		//Initialize final Euclidean metric
		Eigen::MatrixXd fmetric = Eigen::MatrixXd::Identity(dim_output,dim_output);
		
		//Compute the pullback of the final Euclidean metric
		for (int j = num_layers-1; j > -1; j--)
		{
			fpoint = std::move(predict_to_layer(j-1,ipoint));
			jacobian = std::move(layers[j]->compute_partial_derivatives_wrt_inputs(fpoint));
			jacobian_t = std::move(jacobian.transpose());
			Eigen::MatrixXd new_metric = std::move(jacobian_t*fmetric*jacobian);
			fmetric = std::move(new_metric);
		}
		
		//Find eigenvalues and eigenvectots
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(fmetric);
		Eigen::MatrixXd eigen_vec = std::move(es.eigenvectors());
		
		//The last eigenvector is the one along which the e.c. change more
		Eigen::VectorXd max_eigen = eigen_vec.col(dim_input-1);
	
		//Make sure to always proceed in the same direction
		if (old_eigenv.dot(eigen_vec.col(dim_input-1)) < 0.)
		{
			max_eigen = (-1.)*max_eigen;
		}
		
		//Fix the direction as specified by the user
		if(invert_build == false)
		{
			ipoint = ipoint+delta*max_eigen;
		}
		else
		{
			ipoint = ipoint-delta*max_eigen;
		}

		//Compute the new energy
		energy += delta*(max_eigen.dot(fmetric*max_eigen));
		
		//Print energy and prediction
		std::cout << "Energy: " << energy << endl;
		std::cout << "Prediction: "<< predict(ipoint) << endl;
		std::cout << "POINT:" << endl;
		std::cout << ipoint << endl;
		std::cout << "****" << endl;
		//Save the eigenvector of this iteration to compare with the next one,
		//in order to proceed in the same sense.
		old_eigenv = max_eigen;
		
		//Save the result in the desired file (in csv format)
		for (int i = 0; i < len-1; i++)
		{
			file << ipoint(i);
			file << separating_char;
		}
		file << ipoint(len-1);
		file << "\n";
	}	
}
//------------------------------------------------------------------------------	
double neural_network::loss_mse(const std::vector<Eigen::VectorXd> &data, const std::vector<Eigen::VectorXd> &features)
{
	const int len_data = data.size();
	double loss = 0.;
	for (int i = 0; i < len_data; i++)
	{
		Eigen::VectorXd delta = std::move(predict(data[i])-features[i]);
		loss += norm_squared(delta)/len_data;
	}
	return loss;
}
//------------------------------------------------------------------------------	
double neural_network::accuracy(const std::vector<Eigen::VectorXd> &data, const std::vector<Eigen::VectorXd> &features)
{
	const double epsilon = .1;
	int correct = 0;
	int const total = data.size();
	double result;
	for (int i = 0; i < total; i++)
	{
		Eigen::VectorXd temp = std::move(argmax(predict(data[i]))-features[i]);
		if (norm(temp)<epsilon)
		correct += 1;
	}
	result = (1.*correct)/(1.*total);
	return result;
	
}

//#endif /* FCDNN_CPP */

