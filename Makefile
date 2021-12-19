CC = g++

CFLAGS_CC = -std=c++17 -g -I -O3 -march=native -pipe -flto -lpthread -DNDEBUG -D_GLIBCXX_PARALLEL

CFLAGS_WARNINGS = -Wall -Wextra -Wreturn-type -Wpointer-arith -Wcast-align -fstrict-aliasing -Wno-unused-local-typedefs

CFLAGS_OPENMP = -fopenmp -fopenmp-simd

CFLAGS_OPTIMIZATION = -Ofast -fno-builtin -ffast-math -mmmx -msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2 -mavx -mavx2 -mfma -frename-registers -m64 -ftree-vectorize -funroll-loops

CFLAGS = $(CFLAGS_CC) $(CFLAGS_WARNINGS) $(CFLAGS_OPENMP) $(CFLAGS_OPTIMIZATION)

make: main.o layer.o fc_layer_sg.o fc_layer_sp.o fc_layer_sl.o fc_layer_sm.o neural_network.o activation_functions.o read_csv.o
	$(CC) $(CFLAGS) -o main.x main.o layer.o fc_layer_sg.o fc_layer_sp.o fc_layer_sl.o fc_layer_sm.o neural_network.o activation_functions.o read_csv.o

activation_functions.o: activation_functions.h activation_functions.cpp
	$(CC) $(CFLAGS) -c activation_functions.cpp

layer.o: layer.cpp layer.h activation_functions.o
	$(CC) $(CFLAGS) -c layer.cpp

fc_layer_sg.o: layer.h layer.cpp fc_layer_sg.cpp fc_layer_sg.h activation_functions.o
	$(CC) $(CFLAGS) -c fc_layer_sg.cpp

fc_layer_sp.o: layer.h layer.cpp fc_layer_sp.cpp fc_layer_sp.h activation_functions.o
	$(CC) $(CFLAGS) -c fc_layer_sp.cpp
	
fc_layer_sl.o: layer.h layer.cpp fc_layer_sl.cpp fc_layer_sl.h activation_functions.o
	$(CC) $(CFLAGS) -c fc_layer_sl.cpp
	
fc_layer_sm.o: layer.h layer.cpp fc_layer_sm.cpp fc_layer_sm.h activation_functions.o
	$(CC) $(CFLAGS) -c fc_layer_sm.cpp

neural_network.o: layer.h neural_network.cpp activation_functions.o
	$(CC) $(CFLAGS) -c neural_network.cpp

main.o: main.cpp
	$(CC) $(CFLAGS) -c main.cpp	
 
read_csv.o: read_csv.h read_csv.cpp
	$(CC) $(CFLAGS) -c read_csv.cpp

clean: 
	rm layer.o fc_layer_sg.o fc_layer_sp.o fc_layer_sl.o neural_network.o main.o activation_functions.o read_csv.o main.x
