//------------------------------------------------------------------------------
//
//		VECTOR UTILS
//		Utils to work with Eigen::VectorXd
//
//------------------------------------------------------------------------------

#ifndef VECTORS_UTILS
#define VECTORS_UTILS

//STL libs
#include<vector>
#include<iostream>
#include<string>
#include<fstream>
#include<math.h>

//Eigen libs
#include "Eigen/Dense"

//OpenMP
#include<omp.h>


//------------------------------------------------------------------------------
//	(pseudo)norms of a vector and related functions
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//Compute the square of a Euclidean norm of a vector
inline double norm_squared(const Eigen::VectorXd &vec)
{
	int len = vec.size();
	double sum = 0.;
	for (int i = 0; i < len; i++)
	{
		sum += vec(i)*vec(i);
	}
	return sum;
}
//------------------------------------------------------------------------------
//Compute the Euclidean norm of a vector
inline double norm(const Eigen::VectorXd &vec)
{
	int len = vec.size();
	double sum = 0.;
	for (int i = 0; i < len; i++)
	{
		sum += vec(i)*vec(i);
	}
	sum = sqrt(sum);
	return sum;
}
//------------------------------------------------------------------------------
//Action of a degenerate bilinear form on two vectors
inline double degenerate_bilinear_form(Eigen::VectorXd &v1, Eigen::VectorXd &v2, Eigen::MatrixXd &mat)
{
	Eigen::VectorXd temp = std::move(mat*v2);
	double res = v1.dot(temp);
	return res;
}
//------------------------------------------------------------------------------
//Normalize a vector w.r.t a given non-degenerate bilinear form
inline Eigen::VectorXd normalize(Eigen::VectorXd &point, Eigen::MatrixXd &mat)
{
	Eigen::VectorXd res(point.size());
	double norm = std::move(degenerate_bilinear_form(point, point, mat));
	norm = sqrt(norm);
	res = res / norm;
	return res;
	
}
//------------------------------------------------------------------------------
//Normalize a collection of vectors w.r.t a given degenerate bilinear form
inline std::vector<Eigen::VectorXd> normalize(std::vector<Eigen::VectorXd> &points, Eigen::MatrixXd &mat)
{
	int len = points.size();
	std::vector<Eigen::VectorXd> res(len);
	#pragma omp parallel for
	for (int i = 0; i < len; i++)
	{
		res[i] = std::move(normalize(points[i],mat));
	}
	return res;
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//	Conversion between different formats
//------------------------------------------------------------------------------

//From Eigen::VectorXd to std::vector<double>
inline std::vector<double> vector_eigen_to_std(Eigen::VectorXd &vec)
{
	int size = vec.size();
	std::vector<double> res(size);
	for (int i = 0; i < size; i++)
	{
		res[i] = vec(i);
	}
	return res;
}
//------------------------------------------------------------------------------
//From std::vector<double> to Eigen::VectorXd
inline Eigen::VectorXd vector_std_to_eigen(std::vector<double> &vec)
{
	int size = vec.size();
	Eigen::VectorXd res(size);
	for (int i = 0; i < size; i++)
	{
		res[i] = vec[i];
	}
	return res;
}
//------------------------------------------------------------------------------
//Convert a collection of std::vector<double> to a collection of Eigen::VectorXd
inline std::vector<Eigen::VectorXd> vector_vector_data_to_vector_eigenv(std::vector<std::vector<double>> &data)
{
	int size = data.size();
	std::vector<Eigen::VectorXd> res(size);
	for (int i = 0; i < size; i++)
	{
		res[i] = vector_std_to_eigen(data[i]);
	}
	return res;
}
//------------------------------------------------------------------------------
//Convert a collection of Eigen::VectorXd to a collection of std::vector<double>
inline std::vector<Eigen::VectorXd> vector_double_data_to_vector_eigenv(std::vector<double> &data)
{
	int size = data.size();
	std::vector<Eigen::VectorXd> res(size);
	for (int i = 0; i < size; i++)
	{
		Eigen::VectorXd temp(1);
		temp(0) = data[i];
		res[i] = std::move(temp);
	}
	return res;
}

//------------------------------------------------------------------------------
//	OTHER UTILS
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//Find the max in a vector
inline double find_max(const Eigen::VectorXd &vec)
{
	int len = vec.size();
	double max = vec(0);
	for (int i = 1; i < len; i++)
	{
		if (max < vec(i)) max = vec(i);
	}
	return max;
}
//------------------------------------------------------------------------------
//Find the min in a vector
inline double find_min(const Eigen::VectorXd &vec)
{
	int len = vec.size();
	double min = vec(0);
	for (int i = 1; i < len; i++)
	{
		if (min > vec(i)) min = vec(i);
	}
	return min;
}
//------------------------------------------------------------------------------
//Check if all the entries of a vector are positive
inline bool all_positive(Eigen::VectorXd &vec)
{
	bool res = true;
	int len = vec.size();
	for(int i = 0; i < len; i++)
	{
		if (vec(i) < 0)
		{
			res = false;
			break;
		}
	}
	return res;
}
//------------------------------------------------------------------------------
//Set to zero all the entries less than a given epsilon
inline void round_epsilon(Eigen::VectorXd &vec, double epsilon)
{
	int len = vec.size();
	for(int i = 0; i < len; i++)
	{
		if (abs(vec(i)) < epsilon)
		{
			vec(i) = 0;
		}
	}
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//	GEOMETRY UTILS
//------------------------------------------------------------------------------

//Reflect a vector on the boundaries of (0,1)^n
inline void boundary_reflection(Eigen::VectorXd &vec, Eigen::VectorXd &point, double epsilon)
{
	int len = vec.size();
	for(int i = 0; i < len; i++)
	{
		if (point(i)<epsilon)
		{
			if (vec(i) < epsilon)
			{
				vec(i) = -1.*vec(i);
			}
		}
		if (point(i)>1-epsilon)
		{
			if (vec(i) > epsilon)
			{
				vec(i) = -1.*vec(i);
			}
		}
	}
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//	Read/Write a vector from/to file
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//Write a vector to file
inline void write_eigen_vectorxf_to_file(std::string filename, char separating_char, Eigen::VectorXd &vec)
{
	std::ofstream file;
	file.open(filename);
	int len = vec.size();
	for (int i = 0; i < len-1; i++)
	{
		file << vec(i);
		file << separating_char;
	}
	file << vec(len-1);
	file.close();
}
//------------------------------------------------------------------------------
//Read a vector from file
inline void read_eigen_vectorxd_to_file(std::string filename, char separating_char, Eigen::VectorXd &vec)
{
	std::ifstream file(filename);
	std::string str_read;
	if (file.is_open())
	{
		getline(file,str_read);	
		std::vector<std::string> str_vec;
		int pos = str_read.find(separating_char);
		int i = 1;
		while (pos > 0)
		{
			std::string temp;
			temp = str_read.substr(0, pos);
			str_read.erase(0,pos+1);
			str_vec.push_back(temp);
			i++;
			pos = str_read.find(separating_char);
		}
		str_vec.push_back(str_read);
		vec = std::move(Eigen::VectorXd::Zero(i));
		for (int j = 0; j < i; j++)
		{
			vec(j) = std::__cxx11::stod(str_vec[j]);
		}
	}
	else
	{
		std::cerr << "read_eigen_Eigen::VectorXd_to_file : File not found." << std::endl;
	}
}



#endif /* VECTORS_UTILS */
