#include"read_csv.h"

using namespace std;
using Eigen::VectorXd;

//------------------------------------------------------------------------------
void read_data_from_csv(std::vector<std::vector<double>> &x_data, std::vector<std::string> &y_data, std::string filename, char separating_char, int num_cols, int x_len)
{
	string temp;
	string line;
	std::ifstream file(filename);
	std::vector<string> lines;
	int nlines = 0;
	//Saving the lines of the file in a vector<string>
	if (file.is_open())
	{
		while (getline(file,line))
		{
			lines.push_back(line);
			nlines++;
		}
		file.close();
	}
	//Read content of lines
	std::vector<std::vector<string>> data_s(nlines);
	for (int i = 0; i < nlines; i++)
	{
		int now = 0;
		data_s[i].resize(num_cols);
		for(int j = 0; j < num_cols; j++)
		{
			
			int pos = lines[i].find(separating_char);
			string temp2;
			temp2 = lines[i].substr(now, pos);
			lines[i].erase(0,pos+1);
			data_s[i][j] = temp2;
		}
	}
	//Prepare x_data and y_data
	x_data.resize(nlines);
	y_data.resize(nlines);
	//Store double x-values
	#pragma omp parallel for
	for (int i = 0; i < nlines; i++)
	{
		x_data[i].resize(x_len);
		cout << "OK 1"<< endl;
		for(int j = 0; j < x_len; j++)
		{
			x_data[i][j] = stod(data_s[i][j]);
		}
	}
	
	#pragma omp parallel for
	for (int i = 0; i < nlines; i++)
	{
		y_data[i].resize(num_cols-x_len);
		for(int j = x_len; j < num_cols; j++)
		{
			y_data[i][j-x_len] = stod(data_s[i][j]);
		}
	}
}
//------------------------------------------------------------------------------
void read_data_from_csv(std::vector<std::vector<double>> &x_data, std::vector<std::vector<double>> &y_data, std::string filename, char separating_char, int num_cols, int x_len)
{
	string temp;
	string line;
	ifstream file(filename);
	vector<string> lines;
	int nlines = 0;
	//Saving the lines of the file in a vector<string>
	if (file.is_open())
	{
		
		while (getline(file,line))
		{
			lines.push_back(line);
			nlines++;
		}
		file.close();
	}
	//Read content of lines
	vector<vector<string>> data_s(nlines);
	for (int i = 0; i < nlines; i++)
	{
		int now = 0;
		data_s[i].resize(num_cols);
		for(int j = 0; j < num_cols; j++)
		{
			int pos = lines[i].find(separating_char);
			string temp2;
			temp2 = lines[i].substr(now, pos);
			lines[i].erase(0,pos+1);
			data_s[i][j] = temp2;
		}
	}
	x_data.resize(nlines);
	y_data.resize(nlines);
	

	//Store double x-values
	for (int i = 0; i < nlines; i++)
	{
		x_data[i].resize(x_len);
		#pragma omp parallel for
		for(int j = 0; j < x_len; j++)
		{
			x_data[i][j] = stod(data_s[i][j]);
		}
	}
	//Store y-data

	for (int i = 0; i < nlines; i++)
	{
		y_data[i].resize(num_cols-x_len);
		#pragma omp parallel for
		for(int j = x_len; j < num_cols; j++)
		{
			y_data[i][j-x_len] = stod(data_s[i][j]);
		}
	}

}
//------------------------------------------------------------------------------

void read_data_from_csv(std::vector<Eigen::VectorXd> &x_data, std::vector<Eigen::VectorXd> &y_data, std::string filename, char separating_char, int num_cols, int x_len)
{
	string temp;
	string line;
	ifstream file(filename);
	vector<string> lines;
	int nlines = 0;
	//Saving the lines of the file in a vector<string>
	if (file.is_open())
	{
		
		while (getline(file,line))
		{
			lines.push_back(line);
			nlines++;
		}
		file.close();
	}
	//Read content of lines
	vector<vector<string>> data_s(nlines);
	for (int i = 0; i < nlines; i++)
	{
		int now = 0;
		data_s[i].resize(num_cols);
		for(int j = 0; j < num_cols; j++)
		{
			int pos = lines[i].find(separating_char);
			string temp2;
			temp2 = lines[i].substr(now, pos);
			lines[i].erase(0,pos+1);
			data_s[i][j] = temp2;
		}
	}
	lines.clear();
	//Prepare x_data and y_data
	x_data.resize(nlines);
	y_data.resize(nlines);

	//Store double x-values
	for (int i = 0; i < nlines; i++)
	{
		x_data[i] = VectorXd::Zero(x_len);
		for(int j = 0; j < x_len; j++)
		{
			x_data[i](j) = stod(data_s[i][j]);
		}
	}
	//Store y-data

	for (int i = 0; i < nlines; i++)
	{
		y_data[i] = VectorXd::Zero(num_cols-x_len);
		for(int j = x_len; j < num_cols; j++)
		{
			y_data[i](j-x_len) = stod(data_s[i][j]);
		}
	}
}
//------------------------------------------------------------------------------
void read_double_x_from_csv(std::vector<std::vector<double>> &x_data, std::string filename, char separating_char, int num_cols)
{
	string temp;
	string line;
	ifstream file(filename);
	vector<string> lines;
	int nlines = 0;
	//Saving the lines of the file in a vector<string>
	if (file.is_open())
	{
		
		while (getline(file,line))
		{
			lines.push_back(line);
			nlines++;
		}
		file.close();
	}
	//Read content of lines
	vector<vector<string>> data_s(nlines);
	for (int i = 0; i < nlines; i++)
	{
		int now = 0;
		data_s[i].resize(num_cols);
		for(int j = 0; j < num_cols; j++)
		{
			
			int pos = lines[i].find(separating_char);
			string temp2;
			temp2 = lines[i].substr(now, pos);
			lines[i].erase(0,pos+1);
			data_s[i][j] = temp2;
		}
		
	

	}
	//Prepare x_data
	x_data.resize(nlines);


	//Store double x-values
	for (int i = 0; i < nlines; i++)
	{
		x_data[i].resize(num_cols);
		cout << "OK 1"<< endl;
		#pragma omp parallel for
		for(int j = 0; j < num_cols; j++)
		{
			x_data[i][j] = stod(data_s[i][j]);
		}
	}
	//Store y-data


}
