#include"layer.h"
#include"fc_layer_sg.h"

fc_layer_sg::fc_layer_sg(const std::vector<std::vector<double>> &weights_,const std::vector<double> &biases_): layer(weights_,biases_)
{
	type = "FC_LAYER_SG";
	input_dim = weights.cols();
	output_dim = weights.rows();
}
//------------------------------------------------------------------------------
fc_layer_sg::fc_layer_sg(const Eigen::MatrixXd &weights_,const std::vector<double> &biases_): layer(weights_,biases_)
{
	type = "FC_LAYER_SG";
	input_dim = weights.cols();
	output_dim = weights.rows();
}
//------------------------------------------------------------------------------
fc_layer_sg::fc_layer_sg(const Eigen::MatrixXd &weights_,const Eigen::VectorXd &biases_): layer(weights_,biases_)
{
	type = "FC_LAYER_SG";
	input_dim = weights.cols();
	output_dim = weights.rows();
}
//------------------------------------------------------------------------------
Eigen::VectorXd fc_layer_sg::predict(const Eigen::VectorXd & input)
{
	Eigen::VectorXd temp;
	temp = std::move(weights*input);
	temp += biases;
	temp = std::move(sigmoid(temp,exp_max_cut,exp_zero_approx));
	return temp;
}
//------------------------------------------------------------------------------
std::vector<Eigen::VectorXd> fc_layer_sg::predict_batch(const std::vector<Eigen::VectorXd> & input)
{
	int len = input.size();
	std::vector<Eigen::VectorXd> result(len);
	
	#pragma omp parallel for num_threads(len) shared(weights,biases,result)
	for (int i = 0; i < len; i++)
	{
		Eigen::VectorXd temp;
		temp = std::move(weights*input[i]);
		temp += std::move(temp+biases);
		temp = std::move(sigmoid(temp,exp_max_cut,exp_zero_approx));
		result[i] = std::move(temp);
	}
	return result;	
}
//------------------------------------------------------------------------------
Eigen::MatrixXd fc_layer_sg::compute_partial_derivatives_wrt_inputs(Eigen::VectorXd &input) 
{
	const int nrows = weights.rows();
	const int ncols = weights.cols();
	Eigen::MatrixXd res(nrows,ncols);
	Eigen::VectorXd temp;
	temp = std::move(weights*input);
	temp += biases;
	for (int i = 0; i < nrows; i++)
	{
		double x;
		if (temp(i)> -1.*exp_max_cut)
			x = exp(-1.*temp(i));
		else
			x = exp_zero_approx;
		#pragma omp for simd
		for (int j = 0; j < ncols; j++)
		{
			res(i,j) = weights(i,j)*(x/(1.+2.*x+x*x));
		}
	}
	return res;
}
//------------------------------------------------------------------------------
Eigen::MatrixXd fc_layer_sg::compute_partial_derivatives_wrt_weights_biases(Eigen::VectorXd &input)
{
	const int nrows = weights.rows();
	Eigen::VectorXd temp;
	temp = std::move(weights*input);
	temp += biases;
	//Numer of variables of weights and biases
	const int w_b_per_row = weights.cols()+1;
	Eigen::MatrixXd res(nrows,w_b_per_row);
	for (int i = 0; i < nrows; i++)
	{
		double x;
		if (temp(i)> -1.*exp_max_cut) x = exp(-1.*temp(i));
		else x = exp_zero_approx;
		#pragma omp for simd
		for(int j = 0; j < w_b_per_row-1; j++)
		{
			res(i,j) = input(j)*x/(1.+2.*x+x*x);
		}
		res(i,w_b_per_row-1) = x/(1.+2.*x+x*x);
	}
	return res;
}
