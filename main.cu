#include "net.h"

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <fstream>
//#include <cstdint>
#include <string>
#include <cstring>
#include <cstdio>
#include <ctime>



using namespace std;

vector<double> inputVals;
vector<double> resultVals;
vector<double> targetVals;

Net myNet;

int reverseInt (int i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void train(int amount){
	

    ifstream file ("data/train-images-idx3-ubyte");
	ifstream file2 ("data/train-labels-idx1-ubyte");
	
	
	
    int magic_number=0;
    int number_of_images=0;
    int n_rows=0;
    int n_cols=0;
    file.read((char*)&magic_number,sizeof(magic_number)); 
    magic_number= reverseInt(magic_number);
    file.read((char*)&number_of_images,sizeof(number_of_images));
    number_of_images= reverseInt(number_of_images);
    file.read((char*)&n_rows,sizeof(n_rows));
    n_rows= reverseInt(n_rows);
    file.read((char*)&n_cols,sizeof(n_cols));
    n_cols= reverseInt(n_cols);
	
	
    int magic_number2=0;
    int number_of_images2=0;
	
	file2.read((char*)&magic_number2,sizeof(magic_number2)); 
	magic_number2= reverseInt(magic_number2);
	
    file2.read((char*)&number_of_images2,sizeof(number_of_images2));
    number_of_images2= reverseInt(number_of_images2);
	
	//cout<<number_of_images<<endl;
	
    for(int i=0;i<amount;++i)
    {
		
		//cout<< "Training Image: "<<i<<endl;
		
		
		inputVals.clear();
		
		
        for(int r=0;r<28;++r)
        {
            for(int c=0;c<28;++c)
            {
                unsigned char temp=0;
				
				
				
                file.read((char*)&temp,sizeof(temp));
				
				//cout<< "Pixel Value: "<< (int)temp<<endl;
				
				double in = ((double)(int)temp) / 255.0;
				//pixelarray.push_back(in);
				
				
	
				in*=2.0;
				in-=1.0;
				
				
				
				//cout<< "Normalized Pixel: "<< in<<endl;
				inputVals.push_back(in);
				
            }
        }
		
		myNet.feedForward(inputVals);
		
		
        unsigned char label=0;
        file2.read((char*)&label,sizeof(label));
		
		
		//cout<< "Image Label: "<< (int)label<<endl;
		
		for(int x= 0; x<10; x++){
			
			targetVals.push_back(-1.0);
			
		}
		
		targetVals[(int)label] = 1.0;
		
		/*
		myNet.getResults(resultVals);
		
		for(int x= 0; x<10; x++){
			
			cout<< x <<" : " << resultVals[x] << endl;
			
		}
		resultVals.clear();
		*/
		
		myNet.backProp(targetVals);
		
		targetVals.clear();
		

		
		//myNet.printAverageError();
		
    }
	
	
}

int test(){
	
    ifstream train ("data/t10k-images-idx3-ubyte");
	ifstream trainlabel ("data/t10k-labels-idx1-ubyte");
	
    int magic_number=0;
    int number_of_images=0;
    int n_rows=0;
    int n_cols=0;
    int magic_number2=0;
    int number_of_images2=0;
		
		
        train.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);
        train.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        train.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        train.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);
		
		trainlabel.read((char*)&magic_number2,sizeof(magic_number2)); 
		magic_number2= reverseInt(magic_number2);
		
        trainlabel.read((char*)&number_of_images2,sizeof(number_of_images2));
        number_of_images2= reverseInt(number_of_images2);
		

		
		int error = 0;
		
		for(int i = 0; i<10000; i++){
			
			
			inputVals.clear();
            for(int r=0;r<28;++r)
            {
                for(int c=0;c<28;++c)
                {
                    unsigned char temp=0;
					
					
					
                    train.read((char*)&temp,sizeof(temp));
					
					//cout<< "Pixel Value: "<< (int)temp<<endl;
					
					double in = ((double)(int)temp) / 255.0;
					
					
					in*=2.0;
					in-=1.0;
					
					
					
					//cout<< "Normalized Pixel: "<< in<<endl;
					inputVals.push_back(in);
					
                }
            }
			
			myNet.feedForward(inputVals);
			myNet.getResults(resultVals);
			
            unsigned char label=0;
            trainlabel.read((char*)&label,sizeof(label));
			
			//max result
			double maxr = resultVals[0];
			int maxindex = 0;
			for(int x= 1; x<10; x++){
				
				if(resultVals[x]>maxr){
					maxr= resultVals[x];
					maxindex = x;
				}
				
			}
			
			if(((int)label) != maxindex){
				error++;
			}

	
		}
		
		
		return error;
	
}


void testingFunc(){
	
	printf("Should be One: %f\n\n",myNet.m_layers[0][4].m_outputWeights[6].weight);
	myNet.allocmemGPU();
	double *invals = new double[784];
	inputVals.clear();
	resultVals.clear();
	
	for(int i=0;i<784;++i){
		invals[i]=1.0;
		inputVals.push_back(1.0);
	}
	
	
	myNet.feedForward(inputVals);
	
	myNet.getResults(resultVals);
	
	printf("Size %lu\n",resultVals.size());
	
	for(int i=0; i<resultVals.size(); i++){
		
		printf("%f\n", resultVals[i]);
	}

	printf("allocating GPU NET\n");
	
	
	
	
	//myNet.feedForward(inputVals);
	printf("Feeding forward in para\n");
	myNet.feedForwardParallel(invals);
	
	printf("Copying Net To CPU\n");
	myNet.copyGpuToCpu();
	
	printf("Dealloc Net\n");
	myNet.deallocmemGPU();
	
	resultVals.clear();
	
	myNet.getResults(resultVals);
	
	printf("\n\n\n");
	
	for(int i=0; i<10; i++){
		printf("%f\n",resultVals[i]);
	}


}


void testingFunc2(){
	
	printf("Should be One: %f\n\n",myNet.m_layers[0][4].m_outputWeights[6].weight);
	myNet.allocmemGPU();
	double *invals = new double[784];
	inputVals.clear();
	resultVals.clear();
	
	for(int i=0;i<784;++i){
		invals[i]=1.0;
		inputVals.push_back(1.0);
	}
	
	
	myNet.feedForward(inputVals);
	myNet.feedForwardParallel(invals);
	
	myNet.getResults(resultVals);
	
	myNet.getResultsFromGPU();
	
	
	printf("CPU\n");
	for(int i=0; i<resultVals.size(); i++){
		
		printf("%f\n",resultVals[i]);
	}
	printf("GPU\n");
	for(int i=0; i<resultVals.size(); i++){
		
		printf("%f\n",myNet.results_h[i]);
	}
	
	double *tvals = new double[10];
	for(int x= 0; x<10; x++){
		
		targetVals.push_back(-1.0);
		tvals[x]=-1.0;
	}
	
	targetVals[2] = 1.0;
	tvals[2] = 1.0;
	
	printf("Backpropigating\n");
	myNet.backProp(targetVals);
	myNet.backPropParallel(tvals);
	
	
	
	resultVals.clear();
	
	myNet.feedForward(inputVals);
	myNet.feedForwardParallel(invals);
	
	myNet.getResults(resultVals);
	
	myNet.getResultsFromGPU();
	
	printf("CPU\n");
	for(int i=0; i<resultVals.size(); i++){
		
		printf("%f\n",resultVals[i]);
	}
	printf("GPU\n");
	for(int i=0; i<resultVals.size(); i++){
		
		printf("%f\n",myNet.results_h[i]);
	}
	
	
	
	myNet.deallocmemGPU();
	
	
}


void train_feed_on_gpu(int amount){
	

    ifstream file ("data/train-images-idx3-ubyte");
	ifstream file2 ("data/train-labels-idx1-ubyte");
	
	
	
    int magic_number=0;
    int number_of_images=0;
    int n_rows=0;
    int n_cols=0;
    file.read((char*)&magic_number,sizeof(magic_number)); 
    magic_number= reverseInt(magic_number);
    file.read((char*)&number_of_images,sizeof(number_of_images));
    number_of_images= reverseInt(number_of_images);
    file.read((char*)&n_rows,sizeof(n_rows));
    n_rows= reverseInt(n_rows);
    file.read((char*)&n_cols,sizeof(n_cols));
    n_cols= reverseInt(n_cols);
	
	
    int magic_number2=0;
    int number_of_images2=0;
	
	file2.read((char*)&magic_number2,sizeof(magic_number2)); 
	magic_number2= reverseInt(magic_number2);
	
    file2.read((char*)&number_of_images2,sizeof(number_of_images2));
    number_of_images2= reverseInt(number_of_images2);
	
	//cout<<number_of_images<<endl;
	
	double *invals = new double[784];
	
    for(int i=0;i<amount;++i)
    {
		
		//cout<< "Training Image: "<<i<<endl;
		
		for(int x=0;x<784;x++)
		{
			invals[x]=0.0;
		}
		
		
        for(int r=0;r<28;++r)
        {
            for(int c=0;c<28;++c)
            {
                unsigned char temp=0;
				
				
				
                file.read((char*)&temp,sizeof(temp));
				
				//cout<< "Pixel Value: "<< (int)temp<<endl;
				
				double in = ((double)(int)temp) / 255.0;
				//pixelarray.push_back(in);
				
				
	
				in*=2.0;
				in-=1.0;
				
				
				
				//cout<< "Normalized Pixel: "<< in<<endl;
				invals[(28*r)+c] = in;
				
            }
        }
		myNet.allocmemGPU();	

		myNet.feedForwardParallel(invals);
		
		myNet.copyGpuToCpu();
		
		myNet.deallocmemGPU();
		
        unsigned char label=0;
        file2.read((char*)&label,sizeof(label));
		
		
		//cout<< "Image Label: "<< (int)label<<endl;
		
		for(int x= 0; x<10; x++){
			
			targetVals.push_back(-1.0);
			
		}
		
		targetVals[(int)label] = 1.0;
		
		/*
		myNet.getResults(resultVals);
		
		for(int x= 0; x<10; x++){
			
			cout<< x <<" : " << resultVals[x] << endl;
			
		}
		resultVals.clear();
		*/
		
		myNet.backProp(targetVals);
		
		targetVals.clear();
		

		
		//myNet.printAverageError();
		
    }
	
	
}




double *tvals = new double[10];

void train_on_gpu(int amount){
	

    ifstream file ("data/train-images-idx3-ubyte");
	ifstream file2 ("data/train-labels-idx1-ubyte");
	
	
	
    int magic_number=0;
    int number_of_images=0;
    int n_rows=0;
    int n_cols=0;
    file.read((char*)&magic_number,sizeof(magic_number)); 
    magic_number= reverseInt(magic_number);
    file.read((char*)&number_of_images,sizeof(number_of_images));
    number_of_images= reverseInt(number_of_images);
    file.read((char*)&n_rows,sizeof(n_rows));
    n_rows= reverseInt(n_rows);
    file.read((char*)&n_cols,sizeof(n_cols));
    n_cols= reverseInt(n_cols);
	
	
    int magic_number2=0;
    int number_of_images2=0;
	
	file2.read((char*)&magic_number2,sizeof(magic_number2)); 
	magic_number2= reverseInt(magic_number2);
	
    file2.read((char*)&number_of_images2,sizeof(number_of_images2));
    number_of_images2= reverseInt(number_of_images2);
	
	//cout<<number_of_images<<endl;
	
	double *invals = new double[784];
	
    for(int i=0;i<amount;++i)
    {
	
		//cout<< "Training Image: "<<i<<endl;

		for(int x=0;x<784;x++)
		{
			invals[x]=0.0;
		}
		

        for(int r=0;r<28;++r)
        {
            for(int c=0;c<28;++c)
            {
                unsigned char temp=0;
				
				
				
                file.read((char*)&temp,sizeof(temp));
				
				//cout<< "Pixel Value: "<< (int)temp<<endl;
				
				double in = ((double)(int)temp) / 255.0;
				//pixelarray.push_back(in);
				
				
	
				in*=2.0;
				in-=1.0;
				
				
				
				//cout<< "Normalized Pixel: "<< in<<endl;
				invals[(28*r)+c] = in;
				
            }
        }
		
		myNet.feedForwardParallel(invals);

        unsigned char label=0;
        file2.read((char*)&label,sizeof(label));
		
		
		//cout<< "Image Label: "<< (int)label<<endl;
		

		
		/*
		myNet.getResults(resultVals);
		
		for(int x= 0; x<10; x++){
			
			cout<< x <<" : " << resultVals[x] << endl;
			
		}
		resultVals.clear();
		*/

		
		for(int x= 0; x<10; x++){

			tvals[x]=-1.0;
		}
	
		tvals[(int)label] = 1.0;
		

		
		myNet.backPropParallel(tvals);
	
		//myNet.printAverageError();
		
    }
	
	
}

int test_on_gpu(){
	
    ifstream train ("data/t10k-images-idx3-ubyte");
	ifstream trainlabel ("data/t10k-labels-idx1-ubyte");
	
    int magic_number=0;
    int number_of_images=0;
    int n_rows=0;
    int n_cols=0;
    int magic_number2=0;
    int number_of_images2=0;
	int error = 0;	
		
        train.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);
        train.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        train.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        train.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);
		
		trainlabel.read((char*)&magic_number2,sizeof(magic_number2)); 
		magic_number2= reverseInt(magic_number2);
		
        trainlabel.read((char*)&number_of_images2,sizeof(number_of_images2));
        number_of_images2= reverseInt(number_of_images2);
		

		
		double *invals = new double[784];
		
	
	
	    for(int i=0;i<10000;++i)
	    {
		
			//cout<< "Training Image: "<<i<<endl;
		
			for(int x=0;x<784;x++)
			{
				invals[x]=0.0;
			}
		
		
	        for(int r=0;r<28;++r)
	        {
	            for(int c=0;c<28;++c)
	            {
	                unsigned char temp=0;
				
				
				
	                train.read((char*)&temp,sizeof(temp));
				
					//cout<< "Pixel Value: "<< (int)temp<<endl;
				
					double in = ((double)(int)temp) / 255.0;
					//pixelarray.push_back(in);
				
				
	
					in*=2.0;
					in-=1.0;
				
				
				
					//cout<< "Normalized Pixel: "<< in<<endl;
					invals[(28*r)+c] = in;
				
	            }
	        }
			

			myNet.feedForwardParallel(invals);
			myNet.getResultsFromGPU();
			
            unsigned char label=0;
            trainlabel.read((char*)&label,sizeof(label));
			
			//max result
			double maxr = myNet.results_h[0];
			int maxindex = 0;
			for(int x= 1; x<10; x++){
				
				if(myNet.results_h[x]>maxr){
					maxr= myNet.results_h[x];
					maxindex = x;
				}
				
			}
			
			if(((int)label) != maxindex){
				error++;
			}

	
		}
		
		
		return error;
	
}


int main(int argc, char** argv)
{
	srand (time(NULL));
	vector<unsigned> topology;
	
	topology.push_back(784);
	topology.push_back(2500);
	//topology.push_back(2000);
	//topology.push_back(1500);
	//topology.push_back(1000);
	topology.push_back(500);
	topology.push_back(49);
	topology.push_back(10);
	
	myNet.init(topology);

	myNet.allocmemGPU();
	
	//training feedforward then back propigation

	//vector<double> inputVals;
	//vector<double> targetVals;
	//vector<double> resultVals;
	

	//train_feed_on_gpu(60000);

	//myNet.outputToFile("784-10_1");
	
	/*
	cout<<"The Error before Gpu: " << ((double)test())/ 10000.0 << endl;
	
	myNet.allocmemGPU();	
	myNet.copyGpuToCpu();
	myNet.deallocmemGPU();
	cout<<"The Error After copy to the Gpu and Back: " << ((double)test())/ 10000.0 << endl;
	*/
	
	//cout<<"The Error is: " << ((double)test())/ 10000.0 << endl;
	

	//testingFunc2();
	
	/*
    std::clock_t start;
    double duration;
    start = std::clock();
	*/
	double error;
	
	/*
	duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
	std::cout<<"GPU time for: "<<myNet.m_layers[1].size()-1<<" is: "<< duration <<'\n';
	*/
	

	
		train_on_gpu(60000);
		error = ((double)test_on_gpu())/ 10000.0;
		cout<<"The Error is: " << error << endl;
		myNet.copyGpuToCpu();
		myNet.outputToFile("nets/784-2000-500-49-10_0");
	

		train_on_gpu(60000);
		error = ((double)test_on_gpu())/ 10000.0;
		cout<<"The Error is: " << error << endl;
		myNet.copyGpuToCpu();
		myNet.outputToFile("nets/784-2000-500-49-10_1");
		
		
		train_on_gpu(60000);
		error = ((double)test_on_gpu())/ 10000.0;
		cout<<"The Error is: " << error << endl;
		myNet.copyGpuToCpu();
		myNet.outputToFile("nets/784-2000-500-49-10_2");
	

		train_on_gpu(60000);
		error = ((double)test_on_gpu())/ 10000.0;
		cout<<"The Error is: " << error << endl;
		myNet.copyGpuToCpu();
		myNet.outputToFile("nets/784-2000-500-49-10_3");
		
		
	
		train_on_gpu(60000);
		error = ((double)test_on_gpu())/ 10000.0;
		cout<<"The Error is: " << error << endl;
		myNet.copyGpuToCpu();
		myNet.outputToFile("nets/784-2000-500-49-10_4");
	

		train_on_gpu(60000);
		error = ((double)test_on_gpu())/ 10000.0;
		cout<<"The Error is: " << error << endl;
		myNet.copyGpuToCpu();
		myNet.outputToFile("nets/784-2000-500-49-10_5");
		
		train_on_gpu(60000);
		error = ((double)test_on_gpu())/ 10000.0;
		cout<<"The Error is: " << error << endl;
		myNet.copyGpuToCpu();
		myNet.outputToFile("nets/784-2000-500-49-10_6");
	

		train_on_gpu(60000);
		error = ((double)test_on_gpu())/ 10000.0;
		cout<<"The Error is: " << error << endl;
		myNet.copyGpuToCpu();
		myNet.outputToFile("nets/784-2000-500-49-10_7");
		
		train_on_gpu(60000);
		error = ((double)test_on_gpu())/ 10000.0;
		cout<<"The Error is: " << error << endl;
		myNet.copyGpuToCpu();
		myNet.outputToFile("nets/784-2000-500-49-10_8");
	

		train_on_gpu(60000);
		error = ((double)test_on_gpu())/ 10000.0;
		cout<<"The Error is: " << error << endl;
		myNet.copyGpuToCpu();
		myNet.outputToFile("nets/784-2000-500-49-10_9");
	
	
	
	myNet.deallocmemGPU();
	
	
	cout<<"DONE"<<endl;
	
}


