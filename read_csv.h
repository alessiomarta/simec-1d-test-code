//------------------------------------------------------------------------------
//
//		READ CSV UTILS
//		Methods to read data form a csv file
//
//------------------------------------------------------------------------------

#ifndef READ_CSV
#define READ_CSV

//STD libs
#include<fstream>
#include<iostream> 
#include<string>
#include<vector>

//OpenMP libs
#include"omp.h"

//Eigen libs
#include "Eigen/Dense"
#include "Eigen/Core"
#include "Eigen/Sparse"
#include "Eigen/Eigenvalues" 

//Read data (features and output data) from "filename".csv. Saves the features in x_data and the outputs in y_data. Separating char is the separating char of the csv file; num_cols the total number of columns of the csv file and x_len the length of the features vector
//Read features (numbers) and outputs (strings)
void read_data_from_csv(std::vector<std::vector<double>> &x_data, std::vector<std::string> &y_data, std::string filename, char separating_char, int num_cols, int x_len);
//Read features (numbers) and outputs (numbers)
void read_data_from_csv(std::vector<std::vector<double>> &x_data, std::vector<std::vector<double>> &y_data, std::string filename, char separating_char, int num_cols, int x_len);
//Read features (numbers) and outputs (numbers)
void read_data_from_csv(std::vector<Eigen::VectorXd> &x_data, std::vector<Eigen::VectorXd> &y_data, std::string filename, char separating_char, int num_cols, int x_len);
//Read the features from "filename".csv. Saves the features in x_data. Separating char is the separating char of the csv file; num_cols the total number of columns, correspoding to the length of the features vector
void read_double_x_from_csv(std::vector<std::vector<double>> &x_data, std::string filename, char separating_char, int num_cols);

#endif /* READ_CSV */
