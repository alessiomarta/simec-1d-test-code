//Eigen libs
#include "Eigen/Dense"

//OpenMP
#include<omp.h>

//STL libs
#include<iostream>
#include<vector>
#include<string>
#include <utility>
#include <stdlib.h>    
#include <time.h>   
#include <fstream>

#include <stdlib.h> 
#include<stdio.h>

//Neural Network libs
#include "layers.h"
#include "activation_functions.h"
#include "neural_network.h"
#include "vectors_utils.h"
#include "activation_functions.h"

//Data libs
#include "read_csv.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
*Structure in which we save the settings read from the settings file. \n
*- dataset_file: Name of the file containing the dataset.
*- x_dataset_len: Size of the input variables.
*- x_dataset_len: Size of the output variables.
*- normalize: Whether the dataset must be normalized or not (true/false).
*- weights_file: Name of the file containing the structure of the neural network and its weights/biases.
*- SiMEC_output_file: Name of the file in which the SiMEC algorithms shall write their outputs.
*- n_iterations: Number of iterations.
*- delta: The integration step delta of the SiMEC-1D algorithm.
*- invert_direction: Whether to invert the direction of SiMEC-1D or not (true/false).
*- starting_point_file: The file containing the point for which we build the equivalence class.
*- normalize_starting: If SiMEC-1D or SiMExp-1D are selected, this parameter specifies whethere the dataset must be normalized or not (true/false). If true,
*  the maximum and the minimum of the dataset the starting point comes from must be provided in a max_mix_struct.
*- algo: Algorithm to run: SiMEC-1D / SiMExp-1D / Predict.
*- epsilon: Epsilon of SiMExp-1D algorithm.
*- delta_simexp: Delta of the SiMExp-1D algorithm, namely the maximum distance from the starting point.
*/

struct settings_info
{
	std::string dataset_file;
	int x_dataset_len;
	int y_dataset_len;
	bool normalize;
	std::string weights_file;
	std::string SiMEC_output_file;
	int n_iterations;
	double delta;
	bool invert_direction;
	std::string starting_point_file;
	bool normalize_starting;
	string algo;
	double epsilon;
	double delta_simexp;
};

//---------------------------------------------------------------------

/**
Structure in which we save max and min of the components of the data.
*/

struct max_mix_struct
{
	std::vector<double> x_max;
	std::vector<double> x_min;
	std::vector<double> y_max;
	std::vector<double> y_min;
};
//---------------------------------------------------------------------

/*
Procedure to read the settings of the program and save them in a settings_info struct.
The settings are given in a csv file specified in the argument init_file. 
The argument separating_char is the separating character of the csv file.
*/

void read_settings(std::string init_file, char separating_char, settings_info & settings_stc)
{
	//Temp vars
	std::ifstream file(init_file);
	std::string str_init;
	//Read the init vars written in init_file 
	if (file.is_open())
	{
		getline(file,str_init);
		//Number of arguments in settings file
		const int n_args = 14;
		std::vector<string> data_file(n_args);
		
		int pos = str_init.find(separating_char);
		int i = 0;
		while (pos > 0)
		{
			std::string temp;
			temp = str_init.substr(0, pos);
			str_init.erase(0,pos+1);
			data_file[i] = temp;
			i++;
			pos = str_init.find(separating_char);
		}
		data_file[n_args-1] = str_init;
		i++;
		if (i != n_args)
		{
			std::cerr << "Settings are not formatted correctly." << endl;
			file.close();
			exit(EXIT_FAILURE);
		}
		else
		{
			//Store the settings in settings_stc
			settings_stc.dataset_file = data_file[0];
			settings_stc.x_dataset_len = stoi(data_file[1]);
			settings_stc.y_dataset_len = stoi(data_file[2]);
			if (data_file[3].compare("y") == 0) settings_stc.normalize = true;
			else settings_stc.normalize = false;
			settings_stc.weights_file = data_file[4];
			settings_stc.SiMEC_output_file = data_file[5];
			settings_stc.n_iterations = stoi(data_file[6]);
			settings_stc.delta = stod(data_file[7]);
			if (data_file[8].compare("y") == 0) settings_stc.invert_direction = true;
			else settings_stc.invert_direction = false;
			settings_stc.starting_point_file = data_file[9];			
			if (data_file[10].compare("y") == 0) settings_stc.normalize_starting = true;
			else settings_stc.normalize_starting = false;
			settings_stc.algo = data_file[11];
			if (settings_stc.algo.compare("SiMExp-1D") == 0) settings_stc.epsilon = stod(data_file[12]);
			if (settings_stc.algo.compare("SiMExp-1D") == 0) settings_stc.delta_simexp = stod(data_file[13]);
			file.close();
			//TODO : Write the exception handler for stod and stoi
		 }	
	}
	else
	{
		std::cerr << "Settings file not found." << endl;
		exit(EXIT_FAILURE);
	}
}

//---------------------------------------------------------------------

/*
Function to normalize data. It returns a max_mix_struct containing info about max/min of each component.
*/

max_mix_struct normalize_data(std::vector<Eigen::VectorXd> & x_data, std::vector<Eigen::VectorXd> & y_data, settings_info & settings_stc)
{
	int npoints = x_data.size();
	std::vector<double> x_max(settings_stc.x_dataset_len);
	std::vector<double> x_min(settings_stc.x_dataset_len);
	std::vector<double> y_max(settings_stc.y_dataset_len);
	std::vector<double> y_min(settings_stc.y_dataset_len);
		
	//Look for max/min in components of x_data and y_data
	for (int j = 0; j < settings_stc.x_dataset_len; j++)
	{	
		x_max[j] = 0.;
		x_min[j] = 0.;
		for (int i = 0; i < npoints; i++)
		{
			if (x_max[j] < x_data[i](j)) x_max[j] = x_data[i](j);
			if (x_min[j] > x_data[i](j)) x_min[j] = x_data[i](j);
		}
	}
	
	for (int j = 0; j < settings_stc.y_dataset_len; j++)
	{	
		y_max[j] = 0.;
		y_min[j] = 0.;
		for (int i = 0; i < npoints; i++)
		{
			if (y_max[j] < y_data[i](j)) y_max[j] = y_data[i](j);
			if (y_min[j] > y_data[i](j)) y_min[j] = y_data[i](j);
		}
	}
	
	//Look for constant values
	bool ok_flag = true;
	for (int i = 0; i < settings_stc.x_dataset_len; i++)
	{
		if(x_max[i]-x_min[i] == 0.) ok_flag = false;
	}
	
	for (int i = 0; i < settings_stc.y_dataset_len; i++)
	{
		if(y_max[i]-y_min[i] == 0.) ok_flag = false;
	}
	
	if (ok_flag == true)
	{
		//Normalize
		for (int j = 0; j < settings_stc.x_dataset_len; j++)
		{	
			for (int i = 0; i < npoints; i++)
			{
				x_data[i](j) = (x_data[i](j)-x_min[j])/(x_max[j]-x_min[j]);
			}
		}
		for (int j = 0; j < settings_stc.y_dataset_len; j++)
		{	
			for (int i = 0; i < npoints; i++)
			{
				y_data[i](j) = (y_data[i](j)-y_min[j])/(y_max[j]-y_min[j]);
			}
		}

	}
	else
	{
		cout << "Cannot normalize!" << endl;
	}

	max_mix_struct result;
	result.x_max = move(x_max);
	result.x_min = move(x_min);
	result.y_max = move(y_max);
	result.y_min = move(y_min);
	
	return result;
}

//---------------------------------------------------------------------

/*
Function to normalize a single x-point, taking in input a max_mix_struct containing info about max/min of each component.
*/

Eigen::VectorXd normalize_xpoint(Eigen::VectorXd &x_point, max_mix_struct normalization_info)
{
	Eigen::VectorXd normalized_x_point(x_point.size());
	int len = x_point.size();
	//Normalize
	for (int i = 0; i < len; i++)
	{	
		normalized_x_point(i) = (x_point(i)-normalization_info.x_min[i])/(normalization_info.x_max[i]-normalization_info.x_min[i]);

	}
	return normalized_x_point;
}


//---------------------------------------------------------------------

int main()
{

	settings_info settings;
	read_settings("init/settings", '\t', settings);	
	std::vector<Eigen::VectorXd> x_data;
	std::vector<Eigen::VectorXd> y_data;
	max_mix_struct max_min_storage;
	
	if (settings.dataset_file.compare("#") != 0)
	{
		read_data_from_csv(x_data, y_data, settings.dataset_file, ';', settings.x_dataset_len+settings.y_dataset_len, settings.x_dataset_len);
		cout << "Dataset loaded" << endl;
		if (settings.normalize == true)
		{
			
			max_min_storage = normalize_data(x_data, y_data, settings);
		}
	}
	
	
	neural_network net; 
	net.network_read(settings.weights_file);
	
	Eigen::VectorXd starting_point;
	Eigen::VectorXd H_inf;
	Eigen::VectorXd H_sup;
	
	read_eigen_vectorxd_to_file(settings.starting_point_file, '\t', starting_point);
	read_eigen_vectorxd_to_file("init/h_inf.csv", '\t', H_inf);
	read_eigen_vectorxd_to_file("init/h_sup.csv", '\t', H_sup);
	cout << "Starting point:" << endl;
	cout << starting_point << endl;
	
	cout << "Initial prediction : " << net.predict(starting_point) << endl;
	
	if (settings.normalize_starting == true && settings.dataset_file.compare("#") != 0)
	{
		
		max_min_storage = normalize_data(x_data, y_data, settings);
		cout << "Normalized starting point:" << endl;
		starting_point = move(normalize_xpoint(starting_point,max_min_storage));
		H_inf = move(normalize_xpoint(H_inf,max_min_storage));
		H_sup = move(normalize_xpoint(H_sup,max_min_storage));
		cout << max_min_storage.x_max[1] << endl;
		cout << starting_point << endl;
	}
	
	//int count = 0;
	
	string fname = settings.SiMEC_output_file;
	//Remove the old output file
	remove(fname.c_str());
	
	//If the choice is SiMExp-1D
	if (settings.algo.compare("SiMExp-1D") == 0)
	{
	
		//Temp vars we need to store the original point
		Eigen::VectorXd temp_1 = move(Eigen::VectorXd(starting_point));
		Eigen::VectorXd temp_2 = move(Eigen::VectorXd(starting_point));

		//Where to save the results
		string fname = settings.SiMEC_output_file;
		//Parameters of the algorithm
		double epsilon = settings.epsilon;
		//To avoid infinite loops when changing equivalence classes
		int max_steps = 10000;
		//Divide epsilon in nst part
		int nst = 100;
		for (int n = 0; n < nst; n++)
		{
			net.SiMEC_1D_stop_boundary(fname,settings.n_iterations,settings.delta,true,starting_point,H_inf,H_sup); 
			starting_point = move(Eigen::VectorXd(temp_2));
			net.SiMEC_1D_stop_boundary(fname,settings.n_iterations,settings.delta,false,starting_point,H_inf,H_sup); 
			starting_point = move(Eigen::VectorXd(temp_2));
			cout << starting_point << endl;
			net.SiMExp_1D(fname,settings.delta_simexp,epsilon/(1.*nst),max_steps,true,starting_point); 
			temp_2 = move(Eigen::VectorXd(starting_point));
		}
		starting_point = move(Eigen::VectorXd(temp_1));
		temp_2 = move(Eigen::VectorXd(temp_1));
		for (int n = 0; n < nst; n++)
		{
			net.SiMEC_1D_stop_boundary(fname,settings.n_iterations,settings.delta,true,starting_point,H_inf,H_sup); 
			starting_point = move(Eigen::VectorXd(temp_2));
			net.SiMEC_1D_stop_boundary(fname,settings.n_iterations,settings.delta,false,starting_point,H_inf,H_sup); 
			starting_point = move(Eigen::VectorXd(temp_2));
			cout << starting_point << endl;
			net.SiMExp_1D(fname,settings.delta_simexp,epsilon/(1.*nst),max_steps,false,starting_point); 
			temp_2 = move(Eigen::VectorXd(starting_point));
		}
		cout << "Delta: " << settings.delta << endl;
	}
	
	//If the choice is SiMEC-1D
	if (settings.algo.compare("SiMEC-1D") == 0)
	{
	
		string fname = settings.SiMEC_output_file;
		net.SiMEC_1D(fname,settings.n_iterations,settings.delta,true,starting_point); 
	}
	
	//If the choice is Predict
	if (settings.algo.compare("Predict") == 0)
	{
	
		cout << "Prediction:" << endl;
		cout << net.predict(starting_point) << endl;
		cout << endl;
	}
	
   return 0;

}

