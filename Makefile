#Maakefile for Macs, will need to modify for the net to run on linux



#NVCC_FLAGS  = -O3 -I/usr/local/cuda/include -arch=sm_20
#LD_FLAGS    = -lcudart -L/usr/local/cuda/lib

NVCC_FLAGS  = -O3 -I/usr/local/cuda/include
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib

OBJS = net.o neuron.o main.o 
EXENAME = net


COMPILER = nvcc
COMPILER_OPTS = -c
LINKER = nvcc




default : $(EXENAME)

neuron.o : net.h neuron.h neuron.cu
	$(COMPILER) -c -o $@ neuron.cu $(NVCC_FLAGS)

net.o : net.cu net.h neuron.h
	$(COMPILER) -c -o $@ net.cu $(NVCC_FLAGS)

main.o : main.cu net.h neuron.h 
	$(COMPILER) -c -o $@ main.cu $(NVCC_FLAGS)
	
$(EXENAME) : $(OBJS)
	$(LINKER) $(OBJS) -o $(EXENAME) $(LD_FLAGS)
	
clean:
	-rm -f *.o $(EXENAME)