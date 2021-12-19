#include"layer.h"

layer::layer(const std::vector<std::vector<double>> &weights_,const std::vector<double> &biases_)
{
	int lenc = biases_.size();
	biases = Eigen::VectorXd(lenc);
	int lenr = weights_[0].size();
	weights = Eigen::MatrixXd(lenr,lenc);
	#pragma omp parallel for shared(biases,lenc)
	for (int i = 0; i < lenc; i++)
	{
		biases(i) = biases_[i];
		for (int j = 0; j < lenr; j++)
		{
			weights(i,j) = weights_[i][j];
		}
	}
	type = "GENERIC_LAYER";

}
//------------------------------------------------------------------------------
layer::layer(const std::vector<std::vector<double>> &weights_)
{
	int lenr = weights_.size();
	int lenc = weights_[0].size();
	weights = Eigen::MatrixXd(lenr,lenc);
	#pragma omp parallel for shared(biases,lenc)
	for (int i = 0; i < lenc; i++)
	{
		for (int j = 0; j < lenr; j++)
		{
			weights(i,j) = weights_[i][j];
		}
	}
	type = "GENERIC_LAYER";

}
//------------------------------------------------------------------------------
layer::layer(const Eigen::MatrixXd &weights_)
{
	weights = weights_;
	type = "GENERIC_LAYER";
}
//------------------------------------------------------------------------------
layer::layer(const Eigen::MatrixXd &weights_,const std::vector<double> &biases_)
{
	int len = biases_.size();
	biases = Eigen::VectorXd(len);
	#pragma omp parallel for shared(biases,len)
	for (int i = 0; i < len; i++)
	{
		biases(i) = biases_[i];
	}
	
	weights = Eigen::MatrixXd(weights_.rows(),weights_.cols());
	weights = weights_;
	type = "GENERIC_LAYER";
}
//------------------------------------------------------------------------------
layer::layer(const Eigen::MatrixXd &weights_,const Eigen::VectorXd &biases_)
{
	biases = biases_;
	weights = weights_;
	type = "GENERIC_LAYER";
}
//------------------------------------------------------------------------------
layer::layer(const Eigen::MatrixXd &weights_,const Eigen::VectorXd &biases_, int in_dim, int out_dim)
{
	int len = biases_.size();
	biases = Eigen::VectorXd(len);
	#pragma omp parallel for shared(biases,len)
	for (int i = 0; i < len; i++)
	{
		biases(i) = biases_[i];
	}
	
	weights = Eigen::MatrixXd(weights_.rows(),weights_.cols());
	weights = weights_;
	type = "GENERIC_LAYER";
	input_dim = in_dim;
	output_dim = out_dim;
}
//------------------------------------------------------------------------------
const Eigen::VectorXd layer::get_biases()
{
	return biases;
}
//------------------------------------------------------------------------------
const Eigen::MatrixXd layer::get_weights() 
{
	return weights;
}
//------------------------------------------------------------------------------
double layer::get_weight(int i, int j)
{
	return weights(i,j);
}
//------------------------------------------------------------------------------
double layer::get_bias(int i)
{
	return biases(i);
}
//------------------------------------------------------------------------------
const Eigen::MatrixXd layer::get_weights_biases_as_mat()
{
	int nrows = weights.rows();
	int ncols = weights.cols();
	Eigen::MatrixXd res(nrows,ncols+1);
	#pragma omp parallel for
	for (int i = 0; i < nrows; i++)
	{
		#pragma omp simd
		for (int j = 0; j < ncols; j++)
		{
			res(i,j) = weights(i,j);
		}
		res(i,ncols) = biases(i);
	}
	return res;
}
//------------------------------------------------------------------------------
const Eigen::MatrixXd layer::get_weights_biases_as_vec_col_maj()
{
	int len = weights.rows()*(weights.cols()+1);
	int wrows = weights.rows();
	int wcols = weights.cols();
	Eigen::VectorXd res(len);
	for (int i = 0; i < wrows; i++)
	{
		#pragma omp simd
		for (int j = 0; j < wcols; j++)
		{
			res(i+i*wcols+j) = weights(i,j);
		}
		res(i+i*(wcols)+wcols) = biases(i);
	}
	return res;
}
//------------------------------------------------------------------------------
const Eigen::MatrixXd layer::get_weights_biases_as_vec_row_maj()
{
	const int len = weights.rows()*(weights.cols()+1);
	const int wrows = weights.rows();
	const int wcols = weights.cols();
	Eigen::MatrixXd res(1,len);
	for (int i = 0; i < wrows; i++)
	{
		#pragma omp simd
		for (int j = 0; j < wcols; j++)
		{
			res(0,i+i*wcols+j) = weights(i,j);
		}
		res(0,i+i*(wcols)+wcols) = biases(i);
	}
	return res;
}
//------------------------------------------------------------------------------	
void layer::set_biases(const std::vector<double> & biases_)
{
	int len = biases_.size();
	biases.resize(len);
	#pragma omp parallel for
	for (int i = 0; i < len; i++)
	{
		biases(i) = biases_[i];
	}
}
//------------------------------------------------------------------------------
void layer::set_biases(const Eigen::VectorXd & biases_)
{
	biases = biases_;
}
//------------------------------------------------------------------------------
void layer::set_weights(const Eigen::MatrixXd & weights_)
{
	weights = weights_;
}
//------------------------------------------------------------------------------
void layer::set_weight(int i, int j, double weight_)
{
	weights(i,j) = weight_;
}
//------------------------------------------------------------------------------
void layer::set_bias(int i,double bias_)
{
	biases(i) = bias_;
}
//------------------------------------------------------------------------------
void layer::set_weights_biases(Eigen::MatrixXd &source)
{
	const int nrows = weights.rows();
	const int ncols = weights.cols();
	for (int i = 0; i < nrows; i++)
	{
		#pragma omp simd
		for (int j = 0; j < ncols; j++)
		{
			weights(i,j) = source(0,j+i*(ncols+1));
		}
		biases(i) = source(0,i*(ncols+1)+ncols);
	}
}
//------------------------------------------------------------------------------
void layer::set_weights_biases_row_maj(Eigen::MatrixXd &source)
{
	const int nrows = weights.rows();
	const int ncols = weights.cols();
	for (int i = 0; i < nrows; i++)
	{
		#pragma omp simd
		for (int j = 0; j < ncols; j++)
		{
			weights(i,j) = source(0,j+i*(ncols+1));
		}
		biases(i) = source(0,i*(ncols+1)+ncols);
	}
}
//------------------------------------------------------------------------------
void layer::set_weights_biases_compact(Eigen::MatrixXd &source)
{
	int nrows = weights.rows();
	int ncols = weights.cols();
	for (int i = 0; i < nrows; i++)
	{
		for (int j = 0; j < ncols; j++)
		{
			weights(i,j) = source(i,j);
		}
		biases(i) = source(ncols,i);
	}
}
//------------------------------------------------------------------------------
int layer::get_num_nodes()
{
		return output_dim;
}
//------------------------------------------------------------------------------
int layer::get_input_size()
{
		return input_dim;
}
//------------------------------------------------------------------------------
int layer::get_weights_rows()
{
	return weights.rows();
}
//------------------------------------------------------------------------------
int layer::get_weights_cols()
{
	return weights.cols();
}
//------------------------------------------------------------------------------
std::string layer::get_type()
{
	return type;
}
//------------------------------------------------------------------------------
void layer::transpose_weights()
{
	int nrows = weights.rows();
	int ncols = weights.cols();
	double temp;
	#pragma omp parallel for
	for (int i = 0; i < ncols; i++)
	{
		for (int j = 0; j < nrows; j++)
		{
			temp = weights(i,j);
			weights(i,j) = weights(j,i);
			weights(j,i)=temp;
		}
	}
}
//------------------------------------------------------------------------------
void layer::set_exp_max_cut(double exp_max_cut_)
{
	exp_max_cut = exp_max_cut_;
}
//------------------------------------------------------------------------------
void layer::set_exp_zero_approx(double exp_zero_approx_)
{
	exp_zero_approx = exp_zero_approx_;
}
//------------------------------------------------------------------------------
double layer::get_exp_max_cut()
{
	return exp_max_cut;
}
//------------------------------------------------------------------------------
double layer::get_exp_zero_approx()
{
	return exp_zero_approx;
}
