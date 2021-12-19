/******************************************************
*	
*	ACTIVATION FUNCTIONS FOR FEEDFORWARD LAYERS
*
******************************************************/

/*! 
 *  \brief     Activation functions for feedforward layers.
 *  \details   This class implements the sigmoid, softplus and silu activation functions, along with their derivatives.
 */


#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTION_H

#include<math.h>
#include<vector>
#include<iostream>
#include<omp.h>
#include"Eigen/Dense"


//Simgoid activation function
double sigmoid(double x,double max_exp,double exp_zero_approx);
//First derivative of sigmoid
double sigmoid_derivative(double x,double max_exp,double exp_zero_approx);
//Vectorized sigmoid
Eigen::VectorXd sigmoid(Eigen::VectorXd & x, double & max_exp,double & exp_zero_approx);
std::vector<Eigen::VectorXd> sigmoid(std::vector<Eigen::VectorXd> & x, double & max_exp,double & exp_zero_approx);

//Softplus activation function
double softplus(double x,double max_exp);
//First derivative of softplus
double softplus_derivative(double x,double max_exp);

//SiLu
double silu(double x,double max_exp,double exp_zero_approx);
//SiLu derivative
double silu_derivative(double x,double max_exp,double exp_zero_approx);

//argmax
Eigen::VectorXd argmax(const Eigen::VectorXd &input);

//Normalize the input std::vector wiht softmax
Eigen::VectorXd softmax(const std::vector<double> & input);
//Normalize the input Eigen::VectorXd wiht softmax
Eigen::VectorXd softmax(const Eigen::VectorXd & input);
//Compute the jacobian of the softmax at a point given by a std::vector
Eigen::MatrixXd softmax_jacobian(const std::vector<double> &input);
//Compute the jacobian of the softmax at a point given by a Eigen::VectorXd
Eigen::MatrixXd softmax_jacobian(const Eigen::VectorXd &input);


#endif /* ACTIVATION_FUNCTION_ H*/

