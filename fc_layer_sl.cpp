#include"layer.h"
#include"fc_layer_sl.h"

fc_layer_sl::fc_layer_sl(const std::vector<std::vector<double>> &weights_,const std::vector<double> &biases_): layer(weights_,biases_)
{
	type = "FC_LAYER_SL";
	input_dim = weights.cols();
	output_dim = weights.rows();
}
//------------------------------------------------------------------------------
fc_layer_sl::fc_layer_sl(const Eigen::MatrixXd &weights_,const std::vector<double> &biases_): layer(weights_,biases_)
{
	type = "FC_LAYER_SL";
	input_dim = weights.cols();
	output_dim = weights.rows();
}
//------------------------------------------------------------------------------
fc_layer_sl::fc_layer_sl(const Eigen::MatrixXd &weights_,const Eigen::VectorXd &biases_): layer(weights_,biases_)
{
	type = "FC_LAYER_SL";
	input_dim = weights.cols();
	output_dim = weights.rows();
}
//------------------------------------------------------------------------------
Eigen::VectorXd fc_layer_sl::predict(const Eigen::VectorXd & input)
{
	Eigen::VectorXd temp;
	temp = std::move(weights*input);
	temp += biases;
	int len = temp.size();
	#pragma omp parallel for simd
	for (int i = 0; i < len; i++)
	{
		temp(i) = silu(temp(i),exp_max_cut,exp_zero_approx);
	}
	return temp;
}
//------------------------------------------------------------------------------
std::vector<Eigen::VectorXd> fc_layer_sl::predict_batch(const std::vector<Eigen::VectorXd> & input)
{
	int len = input.size();
	std::vector<Eigen::VectorXd> result(len);
	for (int i = 0; i < len; i++)
	{
		result[i] = std::move(predict(input[i]));
	}
	return result;	
}
//------------------------------------------------------------------------------
Eigen::MatrixXd fc_layer_sl::compute_partial_derivatives_wrt_inputs(Eigen::VectorXd &input) 
{
	const int nrows = weights.rows();
	const int ncols = weights.cols();
	Eigen::MatrixXd res(nrows,ncols);
	Eigen::VectorXd temp;
	temp = std::move(weights*input);
	temp += biases;
	for (int i = 0; i < nrows; i++)
	{
		#pragma omp simd
		for (int j = 0; j < ncols; j++)
		{
			res(i,j) = weights(i,j)*silu_derivative(temp(i),exp_max_cut,exp_zero_approx);	
		}
	}
	return res;
}
//------------------------------------------------------------------------------
Eigen::MatrixXd fc_layer_sl::compute_partial_derivatives_wrt_weights_biases(Eigen::VectorXd &input)
{
	const int nrows = weights.rows();
	Eigen::VectorXd temp;
	temp = std::move(weights*input);
	temp += biases;
	//Numer of variables of weights and biases
	const int w_b_per_row = weights.cols()+1;
	Eigen::MatrixXd res(nrows,w_b_per_row);
	res = std::move(Eigen::MatrixXd::Zero(nrows,w_b_per_row));
	for (int i = 0; i < nrows; i++)
	{
		#pragma omp simd
		for(int j = 0; j < w_b_per_row-1; j++)
		{
			res(i,j) = input(j)*silu_derivative(temp(i),exp_max_cut,exp_zero_approx);
		}
		res(i,w_b_per_row-1) = silu_derivative(temp(i),exp_max_cut,exp_zero_approx);
	}
	return res;
}
