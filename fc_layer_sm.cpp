#include"layer.h"
#include"fc_layer_sm.h"
#include"vectors_utils.h"

fc_layer_sm::fc_layer_sm(const std::vector<std::vector<double>> &weights_, const std::vector<double> &biases_): layer(weights_,biases_)
{
	type = "FC_LAYER_SM";
	input_dim = weights.cols();
	output_dim = weights.rows();
}
//------------------------------------------------------------------------------
fc_layer_sm::fc_layer_sm(const Eigen::MatrixXd &weights_, const Eigen::VectorXd &biases_): layer(weights_,biases_)
{
	type = "FC_LAYER_SM";
	input_dim = weights.cols();
	output_dim = weights.rows();
}
//------------------------------------------------------------------------------
fc_layer_sm::fc_layer_sm(const Eigen::MatrixXd &weights_,const std::vector<double> &biases_): layer(weights_,biases_)
{
	type = "FC_LAYER_SM";
	input_dim = weights.cols();
	output_dim = weights.rows();
}
//------------------------------------------------------------------------------
Eigen::VectorXd fc_layer_sm::predict(const Eigen::VectorXd & input)
{
	Eigen::VectorXd temp;
	temp = std::move(weights*input);
	temp += biases;
	double max = find_max(temp);
	int len = weights.rows();
	Eigen::VectorXd rescaled_inp(len);
	for (int i = 0; i < len; i++)
	{
		rescaled_inp(i) = temp(i)-max;
	}
	temp = softmax(rescaled_inp);
	return temp;
}
//------------------------------------------------------------------------------
Eigen::MatrixXd fc_layer_sm::compute_partial_derivatives_wrt_inputs(Eigen::VectorXd &input)
{
	int nrows = weights.rows();
	int ncols = weights.cols();
	Eigen::MatrixXd res(nrows,ncols);
	Eigen::MatrixXd jac_act(nrows,ncols);
	Eigen::VectorXd temp(nrows);
	temp = std::move(weights*input);
	temp += biases;
	double max = find_max(temp);
	int len = weights.rows();
	Eigen::VectorXd rescaled_inp(len);
	for (int i = 0; i < len; i++)
	{
		rescaled_inp(i) = temp(i)-max;
	}
	jac_act = std::move(softmax_jacobian(rescaled_inp));
	//#pragma omp parallel for
	for (int i = 0; i < nrows; i++)
	{
		#pragma omp simd
		for (int j = 0; j < nrows; j++)
		{
			res(i,j) = jac_act(i,j)*weights(i,j); 
		}
	
	}
	return res;
}
//------------------------------------------------------------------------------
Eigen::MatrixXd fc_layer_sm::compute_partial_derivatives_wrt_weights_biases(Eigen::VectorXd &input)
{
	const int nrows = weights.rows();
	const int ncols = weights.cols()+1;
	Eigen::VectorXd temp;
	temp = std::move(weights*input);
	temp += std::move(temp+biases);
	Eigen::MatrixXd jac_act(nrows,ncols);
	double max = find_max(temp);
	int len = weights.rows();
	Eigen::VectorXd rescaled_inp(len);
	for (int i = 0; i < len; i++)
	{
		rescaled_inp(i) = temp(i)-max;
	}
	jac_act = std::move(softmax_jacobian(rescaled_inp));
	const int w_b_per_row = weights.cols()+1;
	Eigen::MatrixXd res(nrows,ncols);
	//#pragma omp parallel for
	for (int i = 0; i < nrows; i++)
	{
		#pragma omp simd
		for(int j = 0; j < w_b_per_row-1; j++)
		{
			res(i,j) = input(j)*jac_act(i,j);
		}
		res(i,w_b_per_row-1) = 0.;
		for (int j = 0; j < w_b_per_row-1; j++)
		{
			res(i,w_b_per_row-1) += jac_act(i,j);
		}
	}
	return res;
}
//------------------------------------------------------------------------------

