#include "neuron.h"

double Neuron::eta = 0.39; //overall net learning rate
double Neuron::alpha = 0.1; //momentum, multiplier of last deltaWeight


/*New ctor for use with the loader*/
Neuron::Neuron(double outputVal, vector<Connection> outputWeights, unsigned myIndex, double gradient)
{
	m_outputVal = outputVal;
	//m_outputWeights = vector<Connection> newvec(outputWeights);
	// m_outputWeights = outputWeights;
	vector<Connection>::iterator it;
	for(it = outputWeights.begin(); it != outputWeights.end(); it++)
	{
		m_outputWeights.push_back(*it);
	}
	m_myIndex = myIndex;
	m_gradient = gradient;
	/*These may not even have to be set.*/
	//eta = 0.35;
	//alpha = 0.1;
}

void Neuron::updateInputWeights(Layer &prevLayer)
{
	//The wights we need to update are in the connection container
	// in the neurons in preceding layer
	
	for(unsigned n = 0; n < prevLayer.size(); ++n){
		
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
		
		double newDeltaWeight = 
			//individual input , magnified by the gradient and train rate
			eta 
			* neuron.getOutputVal()
			* m_gradient
			//Also add momentum = a fraction of the previos delta weight
			+ alpha
			* oldDeltaWeight;
				
			
		neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
			
	}
	
	
}

double Neuron::sumDOW(const Layer& nextLayer) const
{
	
	double sum = 0.0;
	
	for(unsigned n=0; n<nextLayer.size() - 1; ++n){
		
		sum+=m_outputWeights[n].weight * nextLayer[n].m_gradient;
		
	}
	
	return sum;
}

void Neuron::calcOutputGradients(double targetVal)
{	
	double delta =targetVal - m_outputVal;
	m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
	
}
void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
	double dow = sumDOW(nextLayer);
	m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
	m_gradient /= nextLayer.size();
}



double Neuron::transferFunction(double x){
	
	//output range [-1.0 .. 1.0]
	double val = tanh(x);
	

	//cout<<"Transfer " << val <<endl;
	
		return val;
}



double Neuron::transferFunction2(double x){
	
	//output range [-1.0 .. 1.0]
	double val = 1/(1+pow(e,-x));
	

	//cout<<"Transfer " << val <<endl;
	
		return val;
}



double Neuron::transferFunctionDerivative(double x){
	
	//tanh derivative
	return 1.0 - (x*x);
}
double Neuron::transferFunctionDerivative2(double x){
	
	//tanh derivative
	return (1-transferFunction2(x))*transferFunction2(x);
}


Neuron::Neuron(unsigned numOutputs, unsigned myIndex){
	

	
	for(unsigned c=0; c<numOutputs; ++c){
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = randomWeight();		
	}

	m_myIndex = myIndex;
	
}

void Neuron::feedForward(const Layer &prevLayer){
	
	double sum = 0.0;
	
	//loop through all the previous layers outputs (which are our inputs)
	//include the bias nueron
	
	for(unsigned n=0; n<prevLayer.size(); ++n){
		sum += prevLayer[n].getOutputVal() *
			 prevLayer[n].m_outputWeights[m_myIndex].weight;
	}
	//cout<<"The sum is"<<sum<<endl;
	
	
	//Code I added

	sum/=(prevLayer.size()/2.0);
	
	
	m_outputVal = Neuron::transferFunction(sum);
	
}