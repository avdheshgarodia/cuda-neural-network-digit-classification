#ifndef NEURON_H
#define NEURON_H


class Neuron;
#include <vector>
#include <cmath>
#include <cstdlib>
#include <string>
#include <cstring>
#include <iostream>
#include <cassert>

#define e  2.71828182845904523536

using namespace std;


typedef vector<Neuron> Layer;

struct Connection
{	
	double weight;
	double deltaWeight;
	
};


class Neuron{

public:
	
	Neuron(unsigned numOutputs, unsigned myIndex);
	Neuron(double outputVal, vector<Connection> outputWeights, unsigned myIndex, double gradient);
	void setOutputVal(double val){ m_outputVal = val; }
	double getOutputVal() const { return m_outputVal; }
	void feedForward(const Layer &prevLayer);
	
	void calcOutputGradients(double targetVal);
	void calcHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);
	void outputWeightsToFile(string filename);
	double m_outputVal;
	vector<Connection> m_outputWeights;
	unsigned m_myIndex;
	double m_gradient;
	
private:	
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	static double transferFunction2(double x);
	static double transferFunctionDerivative2(double x);
	static double randomWeight(void){ 
	
		double ran = rand() / double(RAND_MAX); 
		ran*=2.0;
		ran-=1.0;
		return ran*.11;
		//return 1.0;
	
	}
	double sumDOW(const Layer& nextLayer) const;
	
	
	static double eta;
	static double alpha;

};

#endif