The file 'settings' in 'init' folder, must contain:

- Name of the file containing the dataset (if none, write '#')
- Size of the input variables
- Size of the output variables
- Whether data must be normalized or not ('y'/'n') 
- Name of the file containing the structure of the neural network and its weights/biases
- Name of the file in which the SiMEC algorithms shall write their outputs
- Number of iterations
- Delta of the algorithm
- Whether to invert the direction of SiMEC-1D or not ('y'/'n')
- The file containing the point for which we build the equivalence class
- Algorithm to run: 'SiMEC-1D' / 'SiMExp-1D' / 'Predict'
- Epsilon of SiMExp

All the arguments are separated by a tab character.

----------------------------------------------------------

The file 'starting_point' in 'init' folder contains the coordinates of the starting point, separated by a tab character.

----------------------------------------------------------

If the hypercube of the training points is of the form (a_1,b_1) x (a_2,b_2) x ... x (a_n,b_n), the files 'h_inf.csv' and 'h_sup.csv' contains the n-uples a_1,a_2,...,a_n and b_1,b_2,..._b_n respectively, separated by a tab chracter.

