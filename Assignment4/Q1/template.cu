#include<cuda.h>
#include<bits/stdc++.h>
#include "wb.h"

#define NUM_BINS 4096

#define CUDA_CHECK(ans)                                                   \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
            file, line);
    if (abort)
      exit(code);
  }
}

__global__ void hist(unsigned int* input, unsigned int* bins, int length)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
if(i < length)
{
    atomicAdd(&bins[input[i]], 1);
  }
}

int main(int argc, char *argv[]) {

  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *deviceInput;
  unsigned int *deviceBins;
  wbArg_t args = wbArg_read(argc, argv);
         
  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (unsigned int *)wbImport(wbArg_getInputFile(args, 1),
                                       &inputLength);
  hostBins = (unsigned int *)calloc(NUM_BINS , sizeof(unsigned int));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);
  wbLog(TRACE, "The number of bins is ", NUM_BINS);

  std::cout<<std::endl;
  for(int i=0;i<10;i++)
    std::cout<<hostInput[i]<<" ";
    std::cout<<std::endl;
wbTime_start(GPU, "Allocating GPU memory.");
cudaMalloc((void **)&deviceInput,
             inputLength * sizeof(int));
  cudaMalloc((void **)&deviceBins,
             NUM_BINS * sizeof(int));
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Allocating GPU memory.");
wbTime_start(GPU, "Copying input memory to the GPU.");
cudaMemcpy(deviceInput, hostInput,
             inputLength * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(deviceBins, hostBins,
             NUM_BINS * sizeof(int),
             cudaMemcpyHostToDevice);
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Copying input memory to the GPU.");
 wbLog(TRACE, "Launching kernel");


  wbTime_start(Compute, "Performing CUDA computation");
hist<<<(inputLength-1)/128 + 1, 128>>> (deviceInput, deviceBins, inputLength);
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
 cudaMemcpy(hostBins, deviceBins, NUM_BINS*sizeof(int), cudaMemcpyDeviceToHost);
wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
cudaFree(deviceBins);
  cudaFree(deviceInput);
  wbTime_stop(GPU, "Freeing GPU Memory");
wbSolution(args, hostBins, NUM_BINS);


  free(hostBins);
  free(hostInput);
  return 0;
}
                                                                                                                            88,1          Bot
