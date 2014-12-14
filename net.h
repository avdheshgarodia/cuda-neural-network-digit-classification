#ifndef NET_H
#define NET_H


#include "neuron.h"



typedef vector<Neuron> Layer;



class Net{

public:
	
	//default constructor
	Net();
	//constructor that takes in a certain topology and then builds the net
	Net(const vector<unsigned> &topology);
	//constructor that builds a net from a file
	Net(string filename);
	//initializes a net with a certain topology
	void init(const vector<unsigned> &topology);
	//allocates GPU memory
	void allocmemGPU();
	//frees memory on gpu
	void deallocmemGPU();
	//copies memory from the Gpu to the cpu	
	void copyGpuToCpu();	
	//feeds a vector of inputs through the net
	void feedForward(vector<double> &inputVals);
	//backpropigates on expected target values
	void backProp(const vector<double> &targetVals);\
	//gets the values at the output nuerons
	void getResults(vector<double> &resultVals) const;
	//outputs the net to a file
	int outputToFile(string filename);
	
	
	//parrallel feed forward
	void feedForwardParallel(double * invals);
	//parrallel Back Propigation
	void backPropParallel(double * targetvals);
	//gets the reulst from the GPU and stores it into a
	void getResultsFromGPU();

	//contains the net
	vector<Layer> m_layers; //m_layers[layerNum][neuronNum]
	double * results_h;

private:
	
	double m_error;	
	double m_recentAverageError;
	double m_recentAverageSmoothingFactor;
	int layers;
	

	//device variables to store the net on the GPU
	int * topology_d;
	double * weights_d;
	double * deltaweights_d;
	double * outputval_d;
	double * gradients_d;
	double * results_d;
	int * error_d;
	double * targetvals_d;

};




#endif








