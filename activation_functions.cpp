#include<math.h>
#include<vector>
#include<iostream>
#include<omp.h>
#include"Eigen/Dense"
#include"activation_functions.h"

#pragma omp declare simd
double sigmoid(double x,double max_exp,double exp_zero_approx)
{
	double result;
	if (x > -1.*max_exp) result = 1./(1.+exp(-1.*x));
	else result = exp_zero_approx;
	return result;
}

#pragma omp declare simd
double sigmoid_derivative(double x,double max_exp,double exp_zero_approx)
{
	double result;
	if (x > -1.*max_exp) result = sigmoid(x,max_exp,exp_zero_approx)*(1.-sigmoid(x,max_exp,exp_zero_approx));
	else result = exp_zero_approx;
	return result;
}
//------------------------------------------------------------------------------
Eigen::VectorXd sigmoid(Eigen::VectorXd & x, double & max_exp,double & exp_zero_approx)
{
	int len = x.size();
	Eigen::VectorXd result(len);
	#pragma omp simd
	for (int k = 0; k < len; k++)
	{
		result(k) = sigmoid(x(k), max_exp, exp_zero_approx);
	}
	return result;
		
}
//------------------------------------------------------------------------------
std::vector<Eigen::VectorXd> sigmoid(std::vector<Eigen::VectorXd> & x, double & max_exp,double & exp_zero_approx)
{
	int len = x.size();
	int datum_len = x[0].size();
	std::vector<Eigen::VectorXd> result(len);
	#pragma omp simd collapse(2)
	for (int i = 0; i < len; i++)
	{
		for (int k = 0; k < datum_len; k++)
		{
			result[i](k) = sigmoid(x[i][k], max_exp, exp_zero_approx);
		}
	}
	return result;
		
}
//------------------------------------------------------------------------------
#pragma omp declare simd
double softplus(double x,double max_exp)
{
	double result;
	if (x < max_exp) result = log(1+exp(x));
	else result = x;
	return result;
}
//------------------------------------------------------------------------------
#pragma omp declare simd
double softplus_derivative(double x,double max_exp)
{
	return sigmoid(x,max_exp,0.);
}
//------------------------------------------------------------------------------
#pragma omp declare simd
double silu(double x,double max_exp,double exp_zero_approx)
{
	double result;
	if (x > -1.*max_exp) result = x*sigmoid(x,max_exp,exp_zero_approx);
	else result = exp_zero_approx;
	return result;
}
//------------------------------------------------------------------------------
#pragma omp declare simd
double silu_derivative(double x,double max_exp,double exp_zero_approx)
{
	double result;
	if (x > -1.*max_exp) result = sigmoid(x,max_exp,exp_zero_approx)+x*sigmoid_derivative(x,max_exp,exp_zero_approx);
	else result = exp_zero_approx;
	return result;
}
//------------------------------------------------------------------------------
Eigen::VectorXd argmax(const Eigen::VectorXd &input)
{
	int len = input.size();
	Eigen::VectorXd result = std::move(Eigen::VectorXd::Zero(len));
	double max = input(0);
	int pos_max = 0;
	result(0)=1.;
	for (int i = 1; i < len; i++)
	{
		if (input(i) > max)
		{
			result(pos_max) = 0.;
			result(i) = 1.;
			pos_max = i;
		}
	}
	return result;
}
//------------------------------------------------------------------------------
//Normalize the input std::vector wiht softmax
Eigen::VectorXd softmax(const std::vector<double> & input)
{
	int len = input.size();
	Eigen::VectorXd temp(len);
	double sum = 0.;
	#pragma omp for simd
	for (int i = 0; i < len; i++)
	{
		sum += exp(input[i]);
	}
	#pragma omp for simd
	for (int i = 0; i < len; i++)
	{
		temp[i] = exp(input[i])/(1.*sum);
	}
	return temp;
}
//------------------------------------------------------------------------------
//Normalize the input Eigen::VectorXd wiht softmax
Eigen::VectorXd softmax(const Eigen::VectorXd & input)
{
	int len = input.size();
	Eigen::VectorXd temp(len);
	double sum = 0.;
	#pragma omp simd reduction(+: sum)
	for (int i = 0; i < len; i++)
	{
		sum += exp(input(i));
	}
	#pragma omp for simd
	for (int i = 0; i < len; i++)
	{
		temp(i) = exp(input(i))/(1.*sum);
	}
	return temp;
}
//------------------------------------------------------------------------------
//Compute the jacobian of the softmax at a point given by a std::vector
Eigen::MatrixXd softmax_jacobian(const std::vector<double> &input)
{
	int len = input.size();
	Eigen::MatrixXd res(input.size(),input.size());
	std::vector<double> temp(len);
	#pragma omp for simd
	for (int i = 0; i < len; i++)
	{
		temp[i] = softmax(input)[i];
	}
	for (int i = 0; i < len; i++)
	{
		#pragma omp simd
		for (int j = 0; j < i; j++)
		{
			res(i,j)= -1.*temp[i]*temp[j];
		}
		res(i,i)= temp[i]*(1.-temp[i]);
		#pragma omp simd
		for (int j = i+1; j < len; j++)
		{
			res(i,j) = -1.*temp[i]*temp[j];
		}
	}
	return res;
}
//------------------------------------------------------------------------------
//Compute the jacobian of the softmax at a point given by a Eigen::VectorXd
Eigen::MatrixXd softmax_jacobian(const Eigen::VectorXd &input)
{
	Eigen::MatrixXd res(input.size(),input.size());
	Eigen::VectorXd temp(input.size());
	temp = std::move(softmax(input));
	int len = input.size();
	for (int i = 0; i < len; i++)
	{
		#pragma omp simd 
		for (int j = 0; j < i; j++)
		{
			res(i,j)=-1.*temp(i)*temp(j);
		}
		res(i,i)=temp(i)*(1.-temp(i));
		#pragma omp simd 
		for (int j = i+1; j < len; j++)
		{
			res(i,j) = -1.*temp(i)*temp(j);
		}
	}
	return res;
}
//------------------------------------------------------------------------------
