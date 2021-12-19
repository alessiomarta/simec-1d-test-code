#ifndef FC_LAYER_SP_CPP
#define FC_LAYER_SP_CPP

#include"layer.h"
#include"fc_layer_sp.h"


fc_layer_sp::fc_layer_sp(const std::vector<std::vector<double>> &weights_, const std::vector<double> &biases_): layer(weights_,biases_)
{
	type = "FC_LAYER_SP";
	input_dim = weights.cols();
	output_dim = weights.rows();
}
//------------------------------------------------------------------------------
fc_layer_sp::fc_layer_sp(const Eigen::MatrixXd &weights_, const Eigen::VectorXd &biases_): layer(weights_,biases_)
{
	type = "FC_LAYER_SP";
	input_dim = weights.cols();
	output_dim = weights.rows();
}
//------------------------------------------------------------------------------
fc_layer_sp::fc_layer_sp(const Eigen::MatrixXd &weights_,const std::vector<double> &biases_): layer(weights_,biases_)
{
	type = "FC_LAYER_SP";
	input_dim = weights.cols();
	output_dim = weights.rows();
}
//------------------------------------------------------------------------------
Eigen::VectorXd fc_layer_sp::predict(const Eigen::VectorXd & input)
{
	Eigen::VectorXd temp;
	temp = std::move(weights*input);
	temp += biases;
	#pragma omp for simd
	for (int i = 0; i < output_dim; i++)
	{
		if (temp(i) < exp_max_cut)
			temp(i) = log(1+exp(temp(i)));		
	}
	return temp;
}
//------------------------------------------------------------------------------
std::vector<Eigen::VectorXd> fc_layer_sp::predict_batch(const std::vector<Eigen::VectorXd> & input)
{
	int len = input.size();
	std::vector<Eigen::VectorXd> result(len);
	
	#pragma omp parallel for num_threads(len) shared(weights,biases,result)
	for (int i = 0; i < len; i++)
	{
		Eigen::VectorXd temp;
		temp = std::move(weights*input[i]);
		temp += std::move(temp+biases);
		temp = std::move(predict(temp));
		result[i] = std::move(temp);
	}
	return result;	
}
//------------------------------------------------------------------------------
Eigen::MatrixXd fc_layer_sp::compute_partial_derivatives_wrt_inputs(Eigen::VectorXd &input)
{
	int nrows = weights.rows();
	int ncols = weights.cols();
	Eigen::MatrixXd res(nrows,ncols);
	Eigen::VectorXd temp;
	temp = std::move(weights*input);
	temp += biases;
	Eigen::VectorXd sig_temp(nrows);
	//Compute sigmoid
	#pragma omp simd
	for (int i = 0; i < nrows; i++)
	{
		if (temp(i) > -1.*exp_max_cut)
			sig_temp(i) = 1./(1.+exp(-1.*temp(i)));
		else
			sig_temp(i) = exp_zero_approx;	
			
	}
	//Compute jacobian
	for (int i = 0; i < nrows; i++)
	{
		
		#pragma omp for simd
		for (int j = 0; j < ncols; j++)
		{
			res(i,j) = weights(i,j)*sig_temp(i);
		}
	}
	return res;
}
//------------------------------------------------------------------------------
Eigen::MatrixXd fc_layer_sp::compute_partial_derivatives_wrt_weights_biases(Eigen::VectorXd &input)
{
	int nrows = weights.rows();
	int ncols = weights.cols()+1;
	Eigen::VectorXd temp(biases.size());
	temp = std::move(weights*input);
	temp += biases;
	nrows = biases.size();
	//Numer of variables of weights and biases
	ncols = (weights.cols()+1)*weights.rows();
	int w_b_per_row = weights.cols()+1;
	Eigen::MatrixXd res(nrows,ncols);
	
	Eigen::VectorXd sig_temp(nrows);
	//Compute sigmoid
	#pragma omp simd
	for (int i = 0; i < nrows; i++)
	{
		if (temp(i) > -1.*exp_max_cut)
			sig_temp(i) = 1./(1.+exp(-1.*temp(i)));
		else
			sig_temp(i) = exp_zero_approx;	
			
	}
	//Compute Jacobian
	for (int i = 0; i < nrows; i++)
	{
		#pragma omp for simd
		for(int j = 0; j < w_b_per_row-1; j++)
		{
			res(i,j) = input(j)*sig_temp(i);
		}
		res(i,w_b_per_row-1) = sig_temp(i);
	}
	return res;
}
//------------------------------------------------------------------------------

#endif /* FC_LAYER_SP_CPP */
