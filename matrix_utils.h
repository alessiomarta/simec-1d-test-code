//------------------------------------------------------------------------------
//	
//		MATRIX UTILS
//		Utils to work with Eigen::MatrixXd
//
//------------------------------------------------------------------------------


#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

//Eigen libs
#include"Eigen/Dense"
#include"Eigen/Core"

//OpenMP libs
#include"omp.h"

//------------------------------------------------------------------------------
//	Functions to work with reduced form of the jacobians.
//	The jacobians of the layers map are sparse matrices
//	with the same number of (consecutive) non-zero entries 
//	in each row (See the documentation).
//	The following functions allow to convert 
//	a standard matrix from/to reduced form and implement 
//	mutliplications between different forms of matrices
//------------------------------------------------------------------------------

//Multiplication between a reduced matrix and a standard one
inline Eigen::MatrixXd reduced_standard_mul(Eigen::MatrixXd &red, Eigen::MatrixXd &std)
{
	int rows_red = red.rows();
	int non_null_per_row = rows_red*(red.cols());
	int rows_std = std.rows();
	int cols_std = std.cols();
	Eigen::MatrixXd result(rows_red,cols_std);
	#pragma omp parallel for collapse(2) shared(red,std,result,rows_red,non_null_per_row,rows_std,cols_std)
	for (int i = 0; i < rows_red; i++)
	{
		for (int j = 0; j < cols_std; j++)
		{
			#pragma omp simd
			for (int k = 0; k < rows_std; k++)
			{
				result(i,j) = red(i,k)*std(k*non_null_per_row,j);
			}
		}
	}
	return result;
}
//------------------------------------------------------------------------------
//Multiplication between a standard matrix and a reduced one
inline Eigen::MatrixXd standard_reduced_mul(Eigen::MatrixXd &std, Eigen::MatrixXd &red)
{
	int rows_std = std.rows();
	int cols_std = std.cols();
	int rows_red = red.rows();
	int cols_red = red.cols();
	int non_null_per_row = red.cols();
	Eigen::MatrixXd result(rows_std,cols_red*rows_red);
	result = move(Eigen::MatrixXd::Zero(rows_std,cols_red*rows_red));
	for (int i = 0; i < rows_std; i++)
	{
		for (int j = 0; j < cols_std; j++)
		{
			#pragma omp simd
			for (int k = 0; k < non_null_per_row; k++)
			{
				result(i,j*non_null_per_row+k) = std(i,j)*red(j,k);
			}
		}
	}
	return result;
}
//------------------------------------------------------------------------------
//Conversion from reduced form to standard form
inline Eigen::MatrixXd reduced_to_standard(Eigen::MatrixXd &a)
{
	
	int rows_a = a.rows();
	int cols_a = a.cols();
	int a_non_null_per_row = cols_a;
	int cols_res = rows_a*a_non_null_per_row;
	Eigen::MatrixXd result;
	result = move(Eigen::MatrixXd::Zero(rows_a,cols_res));
	#pragma omp parallel for shared(a,result,rows_a,cols_a,a_non_null_per_row) 
	for (int i = 0; i < rows_a; i++)
	{
		#pragma omp simd
		for (int j = 0; j < cols_a; j++)
		{
			result(i,i*a_non_null_per_row+j) = a(i,j);
		}
		
	}
	//cout << "OK!" << endl;
	return result;
}
//------------------------------------------------------------------------------
//Conversion from standard form to reduced form
inline Eigen::MatrixXd standard_to_reduced(Eigen::MatrixXd &a)
{
	int rows_a = a.rows();
	int cols_a = a.cols();
	int a_non_null_per_row = cols_a/rows_a;
	int cols_res = a_non_null_per_row;
	Eigen::MatrixXd result(rows_a,cols_res);
	#pragma omp parallel for shared(a,result,rows_a,cols_a,a_non_null_per_row,cols_res)
	for (int i = 0; i < rows_a; i++)
	{
		#pragma omp simd
		for (int j = 0; j < cols_res; j++)
		{
			result(i,j) = a(i,i*a_non_null_per_row+j);
		}
	}
	return result;
}
//------------------------------------------------------------------------------


//------------------------------------------------------------------------------
//		Operations on generic matrices
//------------------------------------------------------------------------------

//Compute the Frobenius norm of a matrix
inline double Frobenius_norm(Eigen::MatrixXd &mat)
{
	const int rows = mat.rows();
	const int cols = mat.cols();
	double result = 0.;
	#pragma omp parallel for collapse(2) shared(mat) reduction(+: result)
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{

			result += mat(i,j)*mat(i,j);
		}
	}
	result = sqrt(result);
	return result;
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//		Write a matrix to file
//------------------------------------------------------------------------------

//Write a matrix to file
inline void write_eigen_matrixxf_to_file(string filename, char separating_char, Eigen::MatrixXd &mat)
{
	std::ofstream file;
	file.open(filename);
	const int nrows = mat.rows();
	const int ncols = mat.cols();
	for (int i = 0; i < nrows; i++)
	{
		for (int j = 0; j < ncols; j++)
		{
			file << mat(i,j);
			file << separating_char;
		}
		file << endl;
	}
	file.close();
}
#endif /* MATRIX_UTILS_H */
