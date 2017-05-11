//CUDA GPU Implementation of Image Smoothing and gradient processing 
//Bhallaji Venkatesan and Divya Sampath Kumar
// Compile by nvcc -arch compute_53 -std=c++11 -I ~/NVIDIA_CUDA-8.0_Samples/common/inc/ -o GPU_imageread_smooth GPU_imageread_smooth.cu

#define _DEFINE_DEPRECATED_HASH_CLASSES 0
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
//#include <cutil_inline.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_cuda_drvapi.h>
#include <helper_functions.h>
#include <helper_image.h>
#include <helper_math.h>
#include <helper_string.h>
#include <helper_timer.h>
//#include "Convolution.h"




#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <cmath>
#include <limits>
#include <sstream>
#include <hash_set>
#include <unordered_set>
#include <iterator>
#include <vector>
#define STRONG_EDGE 0xFF
#define NON_EDGE 0x0
#include "bmp.h"

char *BMPInFile = "car.bmp";
char *BMPOutFile = "suppress.bmp";


//computeImageGradient();

//suppressNonmaximums();



//texture<float, 2, cudaReadModeElementType> deviceMatrixTexture;
texture<unsigned char, 2, cudaReadModeElementType> deviceMatrixTexture;
__device__ __constant__ float deviceXGradientMask[9] = {
	-1, 0, 1,
	-2, 0, 2,
	-1, 0, 1
};
__device__ __constant__ float deviceYGradientMask[9] = {
	 1,  2,  1,
	 0,  0,  0,
	-1, -2, -1
};
__device__ __constant__ float deviceGaussianFilterMask[25] ={
	2,  4,  5,  4, 2,
	4,  9, 12,  9, 4,
	5, 12, 15, 12, 5,
	4,  9, 12,  9, 4,
	2,  4,  5,  4, 2
} ;



__global__ void deviceGaussianConvolution(unsigned char * output, int matrixWidth)
{
	int outputRow = blockIdx.y * blockDim.y + threadIdx.y;
	int outputColumn = blockIdx.x * blockDim.x + threadIdx.x;
 
	float accumulator = 0.0;

#pragma unroll
	for(int i = -2; i <= 2; ++i)
	{
		unsigned matrixColumn = outputColumn + i;
#pragma unroll
		for(int j = -2; j <= 2; ++j)
		{
			accumulator += deviceGaussianFilterMask[(2 + i) + (2 + j)] * tex2D(deviceMatrixTexture, matrixColumn, outputRow + j);
		}
	}
	__syncthreads();
	output[outputRow * matrixWidth + outputColumn] = accumulator / 159;
}


__global__ void deviceComputeGradient(unsigned char* outputGradient, unsigned matrixWidth)
{
	int outputRow = blockIdx.y * blockDim.y + threadIdx.y;
	int outputColumn = blockIdx.x * blockDim.x + threadIdx.x;


	// Get gradients
	float xAccumulator = 0.0;
	float yAccumulator = 0.0;

#pragma unroll
	for(int i = -1; i <= 1; ++i)
	{
		unsigned matrixColumn = outputColumn + i;
#pragma unroll
		for(int j = -1; j <= 1; ++j)
		{
			int maskIndex = (1 + i)* 3 + (1 + j);
			//printf("%f\n", tex2D(deviceMatrixTexture, matrixColumn, outputRow + j));
			xAccumulator += deviceXGradientMask[maskIndex] * tex2D(deviceMatrixTexture, matrixColumn, outputRow + j);
			yAccumulator += deviceYGradientMask[maskIndex] * tex2D(deviceMatrixTexture, matrixColumn, outputRow + j);
		}
	}

	int matrixIndex = outputRow * matrixWidth + outputColumn;
	
	// Get gradient magnitude
	outputGradient[matrixIndex] = abs(xAccumulator) + abs(yAccumulator);
}


void computeGradient(unsigned char * inputMatrix, int matrixWidth, unsigned char * outputGradient)
{
	// Create timer.
    //unsigned int timer = 0;
    //CUT_SAFE_CALL(cutCreateTimer(&timer));

	// Compute memory sizes.
	int matrixMemorySize = matrixWidth * matrixWidth * sizeof(unsigned char);
	
	// Set up device arrays.
	cudaArray* deviceMatrixArray = NULL;
	unsigned char* deviceGradient = NULL;
	//unsigned* deviceEdgeDirections = NULL;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
	cudaMallocArray(&deviceMatrixArray, &channelDesc, matrixWidth, matrixWidth);
	cudaMalloc((void**)&deviceGradient, matrixMemorySize);
	//cudaMalloc((void**)&deviceEdgeDirections, matrixWidth * matrixWidth * sizeof(unsigned));

	// Copy inputs to device.
	cudaMemcpyToArray(deviceMatrixArray, 0, 0, inputMatrix, matrixMemorySize, cudaMemcpyHostToDevice);

	// Set up image matrix as a texture.
	deviceMatrixTexture.addressMode[0] = cudaAddressModeClamp;
	deviceMatrixTexture.addressMode[1] = cudaAddressModeClamp;
	cudaBindTextureToArray(deviceMatrixTexture, deviceMatrixArray);

	// Start timer.
	//CUT_SAFE_CALL(cutStartTimer(timer));

	// Do it!
	dim3 dimGrid(matrixWidth / 16, matrixWidth / 16);
	dim3 dimBlock(16, 16);

	deviceGaussianConvolution<<<dimGrid, dimBlock>>>(deviceGradient, matrixWidth);
	cudaMemcpy(outputGradient, deviceGradient, matrixMemorySize, cudaMemcpyDeviceToHost);
	cudaUnbindTexture(deviceMatrixTexture);
	cudaMemcpyToArray(deviceMatrixArray, 0, 0, outputGradient, matrixMemorySize, cudaMemcpyHostToDevice);
	cudaBindTextureToArray(deviceMatrixTexture, deviceMatrixArray);




	deviceComputeGradient<<<dimGrid, dimBlock>>>(deviceGradient, matrixWidth);

	// Check for errors.
	//CUT_CHECK_ERROR("Kernel execution failed!");

	// Copy device result to host.
	cudaMemcpy(outputGradient, deviceGradient, matrixMemorySize, cudaMemcpyDeviceToHost);
	//cudaMemcpy(outputEdgeDirections, deviceEdgeDirections, matrixMemorySize, cudaMemcpyDeviceToHost);	


	cudaFreeArray(deviceMatrixArray);
	cudaFree(deviceGradient);
	//cudaFree(deviceEdgeDirections);
	cudaUnbindTexture(deviceMatrixTexture);

}




void BitMapRead(char *file,struct bmp_header *bmp, struct dib_header *dib, unsigned char **data, unsigned char **palete)
{
	
	size_t palete_size;

	int fd;

	if((fd = open(file, O_RDONLY)) <0)
		FATAL("Open Source");
	if(read(fd, bmp, BMP_SIZE) != BMP_SIZE)
		FATAL("Read BMP Header");
	if(read(fd, dib, DIB_SIZE) != DIB_SIZE)
		FATAL("Read DIB Header");
	assert(dib->bpp ==8);	

	palete_size = bmp->offset - BMP_SIZE - DIB_SIZE;
	if(palete_size > 0) {
		*palete = (unsigned char *)malloc(palete_size);
		int go = read(fd, *palete, palete_size);
		if(go != palete_size) {
			FATAL("Read Palete");
		}	
	}

	*data = (unsigned char *)malloc(dib->image_size);
	if(read(fd, *data, dib->image_size) != dib->image_size)
		//close(fd);
		FATAL("Read Image");
	close(fd);
}

void BitMapWrite(char *file, struct bmp_header *bmp, struct dib_header *dib, unsigned char *data, unsigned char *palete)
{
	size_t palete_size;
	int fd;

	palete_size = bmp->offset - BMP_SIZE - DIB_SIZE;
	

	if((fd = open(file, O_WRONLY | O_CREAT | O_TRUNC,S_IRUSR | S_IWUSR | S_IRGRP)) <0)
		FATAL("Open Destination");

	if(write(fd, bmp, BMP_SIZE) != BMP_SIZE)
		FATAL("Write BMP Header");

	if(write(fd, dib, DIB_SIZE) != DIB_SIZE)
		FATAL("Write DIB Header");
	
	if(palete_size != 0) {
		if(write(fd, palete, palete_size) != palete_size)
			FATAL("Write Palete");
	}

	
	if(write(fd, data, dib->image_size) != dib->image_size)
		FATAL("Write Image");
	

	
	close(fd);

}


int main()
{

//void computeGradient(const float* inputMatrix, int matrixWidth, float* outputGradient)
//deviceGaussianConvolution<<<dimGrid, dimBlock>>>(deviceGradient, matrixWidth);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	struct bmp_header bmp;
	struct dib_header dib;
	unsigned char *palete = NULL;
	unsigned char *data = NULL;
	unsigned char *out = NULL;

	BitMapRead(BMPInFile, &bmp, &dib, &data, &palete);
	out = (unsigned char *)malloc(dib.image_size);
	
	cudaEventRecord(start);
	//Gaussian Smoothening
	computeGradient(data, dib.width, out);
	//convolution(data, out, dib.width, gaussianMask, 5, gaussianMaskWeight);
	BitMapWrite("GPU_Gaussian_Smooth_Gradient.bmp", &bmp, &dib, out, palete);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Elapsed Time for smoothing:%f\n",milliseconds);

}

