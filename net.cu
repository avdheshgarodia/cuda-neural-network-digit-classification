#include "net.h"


#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>


Net::Net(){
	
}

Net::Net(const vector<unsigned> &topology){
	
	init(topology);
}

void Net::init(const vector<unsigned> &topology){
	
	results_h = new double[10];
	
	m_layers.clear();
	
	unsigned numLayers = topology.size();
	layers = topology.size();
	
	for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum){
		m_layers.push_back(Layer());
		
		//Now we fill the layer with nuerons
		//we loop <= since each layer has a bias nueron
		
		unsigned numOutputs = layerNum == topology.size()-1 ? 0 : topology[layerNum+1];
		
		for(unsigned neuronNum = 0; neuronNum<=topology[layerNum]; ++neuronNum){
			//make a new Nueron
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
		}
		//Force the bias nodes's output value to 1.0
		
		m_layers.back().back().setOutputVal(1.0);
		
		
	}
	
}



void Net::allocmemGPU(){
	
	//cudaMalloc((void*) *layers_d, sizeof(int));
	cudaMalloc((void**) &topology_d, sizeof(int)*layers);
	int topology_h[layers];
	
	int osize =0;
	int wsize = 0;
	for(int i=0; i<layers; i++){
		topology_h[i] = m_layers[i].size();
		osize+=m_layers[i].size();
	}
    cudaMemcpy(topology_d,&topology_h, sizeof(int)*layers, cudaMemcpyHostToDevice);
       
	for(int l=0; l<layers; l++){
		for(int n=0;n<topology_h[l];n++){
			wsize+=m_layers[l][n].m_outputWeights.size();
		}	
	}
	
	double *weights_h = new double[wsize];
	double *deltaweights_h = new double[wsize];
	double *outputval_h = new double[osize];
	
	
	int wcounter=0;
	int lcounter=0;
	
	for(int l=0; l<layers; l++){
		
		for(int n=0;n<topology_h[l];n++){
			
			for(int i =0;i<m_layers[l][n].m_outputWeights.size();i++){
				
			weights_h[i+wcounter] = m_layers[l][n].m_outputWeights[i].weight;	
			deltaweights_h[i+wcounter] =  m_layers[l][n].m_outputWeights[i].deltaWeight;
					
			}
			
			wcounter+=m_layers[l][n].m_outputWeights.size();
			outputval_h[lcounter+n]=m_layers[l][n].m_outputVal;	
		}
		lcounter+=topology_h[l];
	
	}
	
	
	cudaMalloc((void**) &targetvals_d, sizeof(double)*10);
	cudaMalloc((void**) &weights_d, sizeof(double)*wsize);
	cudaMalloc((void**) &deltaweights_d, sizeof(double)*wsize);
	cudaMalloc((void**) &outputval_d, sizeof(double)*osize);
	cudaMalloc((void**) &gradients_d, sizeof(double)*osize);
	cudaMalloc((void**) &error_d, sizeof(int));
	cudaDeviceSynchronize();
	
	cudaMemcpy(weights_d,weights_h, sizeof(double)*wsize, cudaMemcpyHostToDevice);
	cudaMemcpy(deltaweights_d,deltaweights_h, sizeof(double)*wsize, cudaMemcpyHostToDevice);
	cudaMemcpy(outputval_d,outputval_h, sizeof(double)*osize, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	
	delete[] weights_h;
	delete[] deltaweights_h;
	delete[] outputval_h;
	
	
}





void Net::deallocmemGPU(){
	
	cudaFree(weights_d);
	cudaFree(deltaweights_d);
	cudaFree(topology_d);
	cudaFree(outputval_d);
	cudaFree(gradients_d);
	cudaFree(error_d);
	cudaFree(targetvals_d);
	
}

void Net::copyGpuToCpu(){
	
	//cudaMemcpy(C_h, C_d, sizeof(float)*n, cudaMemcpyDeviceToHost);
	
	
	int topology_h[layers];
	
	int osize =0;
	
	
	for(int i=0; i<layers; i++){
		
		osize+=m_layers[i].size();
	}
	
	cudaMemcpy(topology_h, topology_d, sizeof(int)*layers, cudaMemcpyDeviceToHost);
	vector<unsigned> topology;
	for(int i=0; i<layers; i++){
		
		topology_h[i]--;
		topology.push_back(topology_h[i]);
		topology_h[i]++;
	}
	
	init(topology);
	
	int wsize = 0;
	
	
	for(int l=0; l<layers; l++){

		for(int n=0;n<topology_h[l];n++){
			
			wsize+=m_layers[l][n].m_outputWeights.size();
			
		}	
	}
	

	
	double *weights_h = new double[wsize];
	double *deltaweights_h = new double[wsize];
	double *outputval_h = new double[osize];
	
	int wcounter=0;
	int lcounter=0;
	
	cudaMemcpy(weights_h,weights_d, sizeof(double)*wsize, cudaMemcpyDeviceToHost);
	cudaMemcpy(deltaweights_h,deltaweights_d, sizeof(double)*wsize, cudaMemcpyDeviceToHost);
	cudaMemcpy(outputval_h,outputval_d, sizeof(double)*osize, cudaMemcpyDeviceToHost);
	
	cudaDeviceSynchronize();

	
	for(int l=0; l<layers; l++){
		
		for(int n=0;n<topology_h[l];n++){
			
			for(int i =0;i<m_layers[l][n].m_outputWeights.size();i++){
			
			
					
			m_layers[l][n].m_outputWeights[i].weight = weights_h[i+wcounter];	
			m_layers[l][n].m_outputWeights[i].deltaWeight = deltaweights_h[i+wcounter];	
			
			}
			
			wcounter+=m_layers[l][n].m_outputWeights.size();
			m_layers[l][n].m_outputVal = outputval_h[lcounter+n];	
		}
		lcounter+=topology_h[l];
	
	}
	
	delete[] weights_h;
	delete[] deltaweights_h;
	delete[] outputval_h;
	
	
}


	

/*Going to take a file, use a vector representation of that file, then create neurons like so.*/
/*File is going to be lengths separated by space, \n\n weights separated by \n\n,- error. */
/*Note: As of 11/18, we haven't implemented the error part yet in the file.*/
/*Returns -1 on error.*/
/*The "loader."*/
Net::Net(string filename)
{
	FILE* fp;
	long fsize;
	//size_t result;
	char* buf;
	fp = fopen(filename.c_str(), "r");

	/*Just some code to allocate and fill a buffer with the file contents.*/
	fseek(fp, 0, SEEK_END);
	fsize = ftell(fp);
	rewind(fp);
	buf = (char*)malloc(fsize*sizeof(char));
	fread(buf, 1, fsize, fp);
	fclose(fp);
	char * initialbuf = buf;
	unsigned numLayers;
	/*This gets the number of layers based on the layout of the file, then creates an appropriate vector based on that size.*/
	// memcpy(&(buf[0]), &numLayers, (buf - &(buf[0])));
	memcpy(&numLayers, buf, sizeof(unsigned));
	// cout << "numLayers:" << numLayers << endl;
	buf += sizeof(unsigned);
	// printf(buf, "test:%d\n", *buf);
	char * layerVals = buf; /*How many elements are in the current layer. This is a pointer to it.*/
	for(int i = 0; i < numLayers; i++)
	{
		buf += sizeof(int); /*Skip past all the layers to where the first piece of actual data is.*/
	}
	// memcpy((buf), &sum, ((buf)+sizeof(uint32_t) - &(buf));
	// buf+= sizeof(uint32_t);
	for(unsigned layerNum = 0; layerNum < numLayers; layerNum++)
	{
		m_layers.push_back(Layer());
		double outputVal;
		int outWeightssize;
		vector<Connection> outputWeights;
		unsigned myindex;
		double gradient;
		int sum;
		memcpy(&sum, layerVals, sizeof(int));
		// cout << "sum:" << sum << endl;
		int counter=0;
		while(counter != sum)
		{
			//Do memcpy + size of what you're trying to copy, + size of char for the space for each thing. 11/18.
			// memcpy(buf, &outputVal, (sizeof(double)));
			memcpy(&outputVal, buf, sizeof(double));
			buf += sizeof(double);
			memcpy(&outWeightssize, buf, sizeof(int));
			buf = buf + sizeof(int);
			// printf("outWeightssize:%d\n", outWeightssize);
			for(int i = 0; i < outWeightssize; i++)
			{
				double tmp;
				outputWeights.push_back(Connection());
				memcpy(&tmp, buf, sizeof(double));
				outputWeights.back().weight = tmp;
				buf = buf + sizeof(double);
				memcpy(&tmp, buf, sizeof(double));
				outputWeights.back().deltaWeight = tmp;
				buf = buf + sizeof(double);
				cout << "Vals:" << "outWeightssize:"<< outWeightssize << " " << counter << " " << outputWeights.back().weight << " " << outputWeights.back().deltaWeight << endl;
			}
			memcpy(&myindex, buf, (sizeof(unsigned)));
			buf = buf + sizeof(unsigned); // Go past myindex + ' '
			memcpy(&gradient, buf, sizeof(double));
			buf = buf + sizeof(double); // Go past gradient + ' '
			m_layers.back().push_back(Neuron(outputVal, outputWeights, myindex, gradient)); // Invoke the constructor made that takes all the values as input. Might not be .back eventually.
			outputWeights.clear();
			// buf = buf + sizeof(unsigned); // Go past the newline.
			// cout << "Into constructor:" << outputVal << " " << myindex << " " << gradient << endl;

			counter++;
		}
		layerVals += sizeof(int);
	}

	// /*Skip to where the first neuron is.*/
	// buf = strstr(buf, "\n\n");
	// buf+=2; //Skip the newlines.


	//i dont really get this part in the below ctor, so ill just continue with what i think is correct.

	free(initialbuf);
}
/*Takes in a filename. Will output num outputs - outputs - error onto the file.*/
/*Returns 0 on success, -1 on error.*/
/*The "saver."*/
int Net::outputToFile(string filename)
{
	FILE* fp;
	/*Assume valid filename*/
	fp = fopen(filename.c_str(), "w");
	if(!fp)
		return -1;

	vector<Layer>::iterator it;
	vector<Neuron>::iterator iter;

	vector<int> neuronSizes;
	uint32_t sum=0;
	//Get the size of all the neuron vectors.
	for(it = m_layers.begin(); it != m_layers.end(); it++)
	{
		sum += it->size();
		neuronSizes.push_back(it->size()); //This isn't used for now. Can probably be used for error-checking later.
	}
	//size of m_layers, then a space, then size of each neurons vector, then two newlines.
	// fprintf(fp, "%zu' '", m_layers.size());
	unsigned n_layers = m_layers.size();
	cout << "Num_layers:" << n_layers << endl;
	fwrite(&n_layers, sizeof(unsigned), 1, fp);
	for(vector<int>::iterator i=neuronSizes.begin(); i!=neuronSizes.end();i++)
	{
		/*Put the size of each neuron vector into the file.*/
		// fprintf(fp, "%d' '", &sum);
		int size = *i;
		printf("size:%d\n", size);
		fwrite(&size, sizeof(int), 1, fp);
	}
	/*Separate the contents with two newlines.*/
	// fprintf(fp, "\n\n");
	//Iterate through layers
	for(it = m_layers.begin(); it != m_layers.end(); it++)
	{
		//Iterate through neurons.
		for(iter = it->begin(); iter != it->end(); iter++)
		{
			//Put the value of the neurons in the file.
			// fprintf(fp, "%F' '", iter->m_outputVal);
			fwrite(&(iter->m_outputVal), sizeof(double), 1, fp);
			// fprintf(fp, "%d' '",iter->m_outputWeights.size()); // size of vector
			int vecsize = iter->m_outputWeights.size();
			fwrite(&vecsize, sizeof(int), 1, fp);
			// int temp123;
			// fseek(fp, -sizeof(int), SEEK_CUR);
			// fread(&temp123, sizeof(int), 1, fp);
			// printf("vecsize:%d\n",temp123);
			// printf("vecsize:%d\n", vecsize);
			for(vector<Connection>::iterator coni=iter->m_outputWeights.begin(); coni!=iter->m_outputWeights.end(); coni++)
			{
				// fprintf(fp, "%F' '%F' '", coni->weight, coni->deltaWeight); // vector contents
				fwrite(&(coni->weight), sizeof(double), 1, fp);
				fwrite(&(coni->deltaWeight), sizeof(double), 1, fp);
			}
			// fprintf(fp, "%u' '", iter->m_myIndex);
			fwrite(&(iter->m_myIndex), sizeof(unsigned), 1, fp);
			// fprintf(fp, "%F' '", iter->m_gradient);
			fwrite(&(iter->m_gradient), sizeof(double), 1, fp);
			/*I don't think these are needed*/
			// fprintf(fp, "%F' '", iter->eta);
			// fprintf(fp, "%F' '", iter->alpha);
			/*Separate each neuron in a layer with a single newline.*/
			// fprintf(fp, "\n");
		}
		//Separate each layer with two newlines.
		// fprintf(fp, "\n\n");
	}
	//Eventually we'll want some error handling here. Otherwise, everything else should be good.

	fclose(fp);
	return 0; //I don't think anything here can really fail.
}


void Net::getResults(vector<double> &resultVals) const {
	
	resultVals.clear();
	
	for(unsigned n = 0; n < m_layers.back().size() - 1; ++n){
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}
	
}



void Net::feedForward(vector<double> &inputVals){
	
	assert(inputVals.size()==m_layers[0].size() - 1);
	

	//Latch the input vals into the input nuerons
	
	for(unsigned i= 0; i<inputVals.size(); ++i){
		m_layers[0][i].setOutputVal(inputVals[i]);
	}
	
	
	//Forward Propigation
	for(unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum){
		
		Layer &prevLayer = m_layers[layerNum - 1];
		
		for(unsigned n=0; n<m_layers[layerNum].size() - 1; ++n){
			m_layers[layerNum][n].feedForward(prevLayer);
		}
	
	}
			
}
__global__ void latch(double * inputvals, double * nueronoutputvals){
	
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(i<784){
		nueronoutputvals[i]=inputvals[i];
	}
	
}


__global__ void feedForwardkernel(double * weights, 
				double * nueronoutputvals,int *topology, int currlayer, int outoffset, int woffset){
	
					
	
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		
		double sum = 0.0;
		
		if(i<(topology[currlayer+1]-1))
		{
				
			for(unsigned n=0; n < topology[currlayer]; ++n){
				
				//printf("Weight off %d\n",woffset);
				sum += nueronoutputvals[outoffset+n] *	weights[woffset + (n*(topology[currlayer+1]-1)) +i];
				//prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;		 
			}
			//printf("Thread %d Had a sum of %f\n",i,sum);
			//__syncthreads();
			sum/=(topology[currlayer]/2.0);
			//printf("out off%d\n",outoffset);
			//__syncthreads();
			nueronoutputvals[outoffset+topology[currlayer]+i]=tanhf(sum);
		}
		
		
}

void Net::feedForwardParallel(double * invals){
	
	double* invals_d;
	
	cudaMalloc((void**) &invals_d, sizeof(double)*784);
	
	//cudaDeviceSynchronize();
	
	cudaMemcpy(invals_d,invals, sizeof(double)*784, cudaMemcpyHostToDevice);
	
	cudaDeviceSynchronize();
	
	dim3 dim_block_latch(256,1,1);
	dim3 dim_grid_latch(4,1,1);
	
	//run a lacth kernel
	latch<<<dim_grid_latch,dim_block_latch>>>(invals_d,outputval_d);
	cudaDeviceSynchronize();
	
	cudaFree(invals_d);

	dim3 dim_block(512,1,1);
	dim3 dim_grid(8,1,1);
	
	int osize = 0;
	int wsize = 0;
	
	for(int i=0;i<layers-1;i++){
		
		dim3 dim_block(512,1,1);
		dim3 dim_grid((int)((m_layers[i+1].size()/512)+1),1,1);
		//printf("Launching forward kernel\n");
		feedForwardkernel<<<dim_grid, dim_block>>>(weights_d, outputval_d ,topology_d, i, osize, wsize);
		cudaDeviceSynchronize();
		osize+=m_layers[i].size();
		wsize+=m_layers[i].size()*(m_layers[i+1].size()-1);
	}	
			
	
}

__global__ void getResultskernel(double * results, int outoffset, double* outputvals){
	
	int tid = threadIdx.x;
	
	if(tid<10){
		results[tid] = outputvals[outoffset+tid];
	}
	
	
}

void Net::getResultsFromGPU(){
	
	//Can be stored so that the this does not need to be computed 
	int osize;
	
	for(int i=0; i<layers-1; i++){
		
		osize+=m_layers[i].size();
	}
	
	cudaMalloc((void**) &results_d, sizeof(double)*10);
	
	
	dim3 dim_block(16,1,1);
	dim3 dim_grid(1,1,1);
	
	getResultskernel<<<dim_grid, dim_block>>>(results_d, osize, outputval_d);
	cudaDeviceSynchronize();
	
	for(int i=0;i<10;i++){
		results_h[i]=0.0;
	}
	
	
	cudaMemcpy(results_h,results_d, sizeof(double)*10, cudaMemcpyDeviceToHost);
	cudaFree(results_d);
}
__global__ void calcOutputGradientskernel(double * targetvals, double * outputvals,double * gradients, int outoffset){
	
	int tid = threadIdx.x;
	
	if(tid<10){
		double delta =targetvals[tid] - outputvals[outoffset+tid];
		gradients[outoffset+tid] = delta * (1.0 - (outputvals[outoffset+tid]*outputvals[outoffset+tid]));
	
	}
}

__global__ void calcHiddenGradientskernel(double * weights,double * gradients, int outoffset,int woffset, int * topology, int currentlayer, double * outputvals){


	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(i < topology[currentlayer]){
	
	
		double dow = 0.0;
		
		for(int n=0; n< topology[currentlayer+1] - 1; ++n){
		
			dow+=weights[woffset + (i*(topology[currentlayer+1]-1)) + n] * gradients[outoffset+topology[currentlayer]+n];
			
		}
		
			
	
		gradients[outoffset+i] = dow * (1.0 - (outputvals[outoffset+i]*outputvals[outoffset+i]));
		gradients[outoffset+i] /= topology[currentlayer+1];
	
	}
	
}	

__global__ void updateInputWeightskernel(double * weights,double * gradients, double* outputvals, int woffset, 
									int outoffset, double * deltaweights, int *topology, int currlayer){


	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;			
	
	if(i < topology[currlayer] - 1){						

	for(int n = 0; n < topology[currlayer-1]; ++n){
		
		//Neuron &neuron = prevLayer[n];
		//double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
		
		double newDeltaWeight = 
			//individual input , magnified by the gradient and train rate
			.39 
			* outputvals[outoffset-topology[currlayer-1]+n]
			* gradients[outoffset+i]
			//Also add momentum = a fraction of the previos delta weight
			+ .1
			* deltaweights[woffset + (n*(topology[currlayer]-1)) +i];
				
			
		deltaweights[woffset + (n*(topology[currlayer]-1)) +i] = newDeltaWeight;
		weights[woffset + (n*(topology[currlayer]-1)) +i] += newDeltaWeight;
			
	}

	}	
}
void Net::backPropParallel(double * targetvals){
	
	cudaMemcpy(targetvals_d,targetvals, sizeof(double)*10, cudaMemcpyHostToDevice);
	
	
	//calcoutput gradients
	
	int osize = 0;
	int wsize = 0;
	
	int osize2 = 0;
	int wsize2 = 0;
	
	for(int i=0; i<layers-1; i++){
		
		osize+=m_layers[i].size();
		//wsize+=m_layers[i].size()*(m_layers[i+1].size()-1);
	}
	
	if(layers>2){
		for(int i=0; i<layers-2; i++){
			wsize+=m_layers[i].size()*(m_layers[i+1].size()-1);
			osize2+=m_layers[i].size();
		}
	}
	
	wsize2=wsize;
	
	dim3 dim_block(16,1,1);
	dim3 dim_grid(1,1,1);
	
	
	calcOutputGradientskernel<<<dim_grid, dim_block>>>(targetvals_d, outputval_d ,gradients_d, osize);
	cudaDeviceSynchronize();
	
	
	
	//calc hidden gradients by going backwords through net
	if(layers>2){
		
		for(int l = layers - 2; l>0; --l){
			
		dim3 dim_block(512,1,1);
		dim3 dim_grid((int)((m_layers[l].size()/512)+1),1,1);
		/*	
		printf("Calc Hidden Kernel Launch\n");	
		printf("The weight offset: %d\n" , wsize2);
		printf("The output offset: %d\n" , osize2);		
		printf("The Current Layer: %d\n" , l);	
		*/	
		calcHiddenGradientskernel<<<dim_grid, dim_block>>>(weights_d,gradients_d,osize2,wsize2,topology_d, l,outputval_d);
		cudaDeviceSynchronize();
		osize2-=m_layers[l-1].size();
		wsize2-=m_layers[l-1].size()*(m_layers[l].size()-1);
		
		}
	}	
	
	
	//update input weights
	for(int l = layers - 1; l>0; --l){
		
		dim3 dim_block(512,1,1);
		dim3 dim_grid((int)((m_layers[l].size()/512)+1),1,1);
		/*
		printf("Update Inout Weights LAunch\n");
		
		printf("The weight offset: %d\n" , wsize);
		printf("The output offset: %d\n" , osize);		
		printf("The Current Layer: %d\n" , l);	
		*/	
		updateInputWeightskernel<<<dim_grid, dim_block>>>(weights_d,gradients_d,outputval_d,wsize, osize, deltaweights_d,topology_d,l);
		cudaDeviceSynchronize();
		osize-=m_layers[l-1].size();
		if(l-2>=0)
		wsize-=m_layers[l-2].size()*(m_layers[l-1].size()-1);
		
	}
	
}


void Net::backProp(const vector<double> &targetVals){
	
	//calculate overall Net error (RMS of output neuron errors)

	
	assert(targetVals.size()==m_layers.back().size()-1);
	
	
	
	Layer &outputLayer = m_layers.back();
	m_error = 0.0;
	
	
	for(unsigned n = 0; n< outputLayer.size() - 1; ++n){
		double delta = targetVals[n] -outputLayer[n].getOutputVal();
		m_error += delta*delta;
	}
	m_error /= outputLayer.size() - 1;
	m_error  = sqrt(m_error);
	
	//Implement a recent average measurement
	
	m_recentAverageError = 
		(m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
			/ (m_recentAverageSmoothingFactor + 1.0);
	

	// Calculate output layer gradients
	
	for(unsigned n = 0; n< outputLayer.size() - 1; ++n){
		outputLayer[n].calcOutputGradients(targetVals[n]);
	
	}
	
	//calculate gradients on all hidden layers
	
	for(unsigned layerNum = m_layers.size() - 2; layerNum>0; --layerNum){
		Layer &hiddenLayer = m_layers[layerNum];
		Layer &nextLayer   = m_layers[layerNum + 1];
		
		for(unsigned n = 0; n<hiddenLayer.size(); ++n){
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
		
		
	}
	
	
	//From all layers from outputs to first hidden layer,
	//update connection weights
	
	
	for(unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum)
	{	
		Layer &layer    = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum - 1];
		
		for(unsigned n=0; n<layer.size() - 1; ++n){
			layer[n].updateInputWeights(prevLayer);
		}
		
	}	
		
	
}
