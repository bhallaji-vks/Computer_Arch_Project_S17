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
#include <chrono>
#define STRONG_EDGE 0xFF
#define NON_EDGE 0x0

#define highThreshold 100.0
#define lowThreshold 90.0
#include "bmp.h"

using namespace std;


std::unordered_set<unsigned int> visitedPixels;

char *BMPInFile = "Cameraman256.bmp";
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


__global__ void deviceComputeGradient(unsigned char* outputGradient, unsigned matrixWidth, unsigned int* outputEdgeDirectionClassifications)
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

	// Determine edge direction
	float edgeDirection = atan2(yAccumulator, xAccumulator) * (180 / 3.14159265) + 180.0;

	// Classify edge directions
	if((edgeDirection >= 22.5 && edgeDirection < 67.5) || (edgeDirection >= 202.5 && edgeDirection < 247.5))
	{
		outputEdgeDirectionClassifications[matrixIndex] = 1;
	}
	else if((edgeDirection >= 67.5 && edgeDirection < 112.5) || (edgeDirection >= 247.5 && edgeDirection < 292.5))
	{
		outputEdgeDirectionClassifications[matrixIndex] = 2;
	}
	else if((edgeDirection >= 112.5 && edgeDirection < 157.5) || (edgeDirection >= 292.5 && edgeDirection < 337.5))
	{
		outputEdgeDirectionClassifications[matrixIndex] = 3;
	}
	else
	{
		outputEdgeDirectionClassifications[matrixIndex] = 0;
	}






}

__global__ void devicesuppressNonmaximums(int width, unsigned int* edgeDirectionClassifications, unsigned char* gradient, int imgsize)
{
	int outputRow = blockIdx.y * blockDim.y + threadIdx.y;
	int outputColumn = blockIdx.x * blockDim.x + threadIdx.x;
	int i= outputRow*width + outputColumn;
	
	int clockwisePerpendicularI;
	int clockwisePerpendicularJ; 
	int clockwisePerpendicularIndex;
	int counterClockwisePerpendicularIndex;

	switch(edgeDirectionClassifications[i])
	{
		case 0:
			clockwisePerpendicularI = outputRow - 1;
			clockwisePerpendicularJ = outputColumn;
			break;
		case 1:
			clockwisePerpendicularI = outputRow - 1;
			clockwisePerpendicularJ = outputColumn + 1;
			break;
		case 2:
			clockwisePerpendicularI = outputRow;
			clockwisePerpendicularJ = outputColumn + 1;
			break;
		case 3:
			clockwisePerpendicularI = outputRow + 1;
			clockwisePerpendicularJ = outputColumn + 1;
			break;
	}
	
	if(clockwisePerpendicularI < 0 || clockwisePerpendicularJ < 0 || clockwisePerpendicularI >= width || clockwisePerpendicularJ >= width)
	{
		clockwisePerpendicularIndex = -1;
	}
	else
	{
		clockwisePerpendicularIndex= clockwisePerpendicularI * width + clockwisePerpendicularJ;
	}
	
	int counterClockwisePerpendicularI;
	int counterClockwisePerpendicularJ;
	
	switch(edgeDirectionClassifications[i])
	{
		case 0:
			counterClockwisePerpendicularI = outputRow + 1;
			counterClockwisePerpendicularJ = outputColumn;
			break;
		case 1:
			counterClockwisePerpendicularI = outputRow + 1;
			counterClockwisePerpendicularJ = outputColumn - 1;
			break;
		case 2:
			counterClockwisePerpendicularI = outputRow;
			counterClockwisePerpendicularJ = outputColumn - 1;
			break;
		case 3:
			counterClockwisePerpendicularI = outputRow - 1;
			counterClockwisePerpendicularJ = outputColumn - 1;
			break;
	}


	if(counterClockwisePerpendicularI < 0 || counterClockwisePerpendicularJ < 0 ||counterClockwisePerpendicularJ >= width || counterClockwisePerpendicularJ >= width)
	{
		counterClockwisePerpendicularIndex = -1;
	}
	else
	{
		counterClockwisePerpendicularIndex= counterClockwisePerpendicularI * width + counterClockwisePerpendicularJ;
	}
	
	float clockwisePerpendicularValue;
	float counterClockwisePerpendicularValue;

	if(clockwisePerpendicularIndex == -1)
	{
		clockwisePerpendicularValue = 0;
	}
	else
	{
		clockwisePerpendicularValue = gradient[clockwisePerpendicularIndex];
				
	}	

			
	if(counterClockwisePerpendicularIndex == -1)
	{
		counterClockwisePerpendicularValue = 0;
	}
	else
	{
		if(counterClockwisePerpendicularIndex < imgsize && counterClockwisePerpendicularIndex >= 0)
		{
			counterClockwisePerpendicularValue = gradient[counterClockwisePerpendicularIndex];
		}
	}
			
	if(gradient[i] <= clockwisePerpendicularValue || gradient[i] <= counterClockwisePerpendicularValue)
	{
		//cout << "\tPixel suppressed." << endl;
		gradient[i] = 0;
			
	}
	else
	{
		//cout << "\tPixel retained." << endl;

	}
	
}

/*
__host__ __device__ void visitNeighbors(int i, int j, unsigned char* gradientImage, unsigned char* outputEdges, int width, int imgsize)
{
	int pixelIndex = i * width + j;
        
	if(i == 0 || j == 0 || i == width - 1 || j == width - 1 || visitedPixels.find(pixelIndex) != visitedPixels.end()  ||gradientImage[pixelIndex] <  lowThreshold)
	{

		visitedPixels.insert(pixelIndex);
		return;
	}

	outputEdges[pixelIndex] =STRONG_EDGE;
	visitedPixels.insert(pixelIndex);

	visitNeighbors(i - 1, j - 1, gradientImage, outputEdges, width, imgsize);
	visitNeighbors(i - 1, j,gradientImage, outputEdges, width, imgsize);
	visitNeighbors(i - 1, j + 1, gradientImage, outputEdges, width, imgsize);
	visitNeighbors(i, j + 1, gradientImage, outputEdges, width, imgsize);
	visitNeighbors(i + 1, j + 1, gradientImage, outputEdges, width, imgsize);
	visitNeighbors(i + 1, j, gradientImage, outputEdges, width, imgsize);
	visitNeighbors(i + 1, j - 1, gradientImage, outputEdges, width, imgsize);
	visitNeighbors(i, j - 1, gradientImage, outputEdges, width, imgsize);
} */


__global__ void devicePerformHysteresis(unsigned char* gradientImage, unsigned char* outputEdges, int width, int imgsize) 
{	
	
	int outputRow = blockIdx.y * blockDim.y + threadIdx.y;
	int outputColumn = blockIdx.x * blockDim.x + threadIdx.x;
	int i= outputRow*width + outputColumn;

	// Mark out borders and all pixels below the high threshold.
	if(gradientImage[i] >= highThreshold)
	{
		//visitedPixels.insert(i);
		outputEdges[i] = STRONG_EDGE;
		//visitNeighbors(outputRow, outputColumn, gradientImage, outputEdges, width, imgsize);
	}
}


void computeGradient(unsigned char * inputMatrix, int matrixWidth, unsigned char * outputGradient, unsigned int* outputEdgeDirections, float *time_smooth, float *time_gradient, float *time_suppression, float *time_hysteresis)
{
	// Create timer.
    //unsigned int timer = 0;
    //CUT_SAFE_CALL(cutCreateTimer(&timer));
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Compute memory sizes.
	int matrixMemorySize = matrixWidth * matrixWidth * sizeof(unsigned char);
	
	// Set up device arrays.
	cudaArray* deviceMatrixArray = NULL;
	unsigned char* deviceGradient = NULL;
	unsigned int* deviceEdgeDirections = NULL;
	unsigned char* deviceHysOutput = NULL;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
	cudaMallocArray(&deviceMatrixArray, &channelDesc, matrixWidth, matrixWidth);
	cudaMalloc((void**)&deviceGradient, matrixMemorySize);
	cudaMalloc((void**)&deviceHysOutput, matrixMemorySize);
	cudaMalloc((void**)&deviceEdgeDirections, matrixWidth * matrixWidth * sizeof(unsigned int));

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
	cudaEventRecord(start);
	deviceGaussianConvolution<<<dimGrid, dimBlock>>>(deviceGradient, matrixWidth);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(time_smooth, start, stop);
	cudaMemcpy(outputGradient, deviceGradient, matrixMemorySize, cudaMemcpyDeviceToHost);
	cudaUnbindTexture(deviceMatrixTexture);
	cudaMemcpyToArray(deviceMatrixArray, 0, 0, outputGradient, matrixMemorySize, cudaMemcpyHostToDevice);
	cudaBindTextureToArray(deviceMatrixTexture, deviceMatrixArray);



	cudaEventRecord(start);
	deviceComputeGradient<<<dimGrid, dimBlock>>>(deviceGradient, matrixWidth, deviceEdgeDirections);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(time_gradient, start, stop);
	// Check for errors.
	//CUT_CHECK_ERROR("Kernel execution failed!");

	// Copy device result to host.
	cudaMemcpy(outputGradient, deviceGradient, matrixMemorySize, cudaMemcpyDeviceToHost);
	cudaMemcpy(outputEdgeDirections, deviceEdgeDirections, matrixMemorySize, cudaMemcpyDeviceToHost);	
	
	cudaEventRecord(start);
	devicesuppressNonmaximums<<<dimGrid, dimBlock>>>(matrixWidth, deviceEdgeDirections, deviceGradient, matrixMemorySize);
	cudaMemcpy(outputGradient, deviceGradient, matrixMemorySize, cudaMemcpyDeviceToHost);	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(time_suppression, start, stop);
	
	cudaEventRecord(start);
	devicePerformHysteresis<<<dimGrid, dimBlock>>>(deviceGradient,deviceHysOutput, matrixWidth, matrixMemorySize);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(time_hysteresis, start, stop);
	cudaMemcpy(outputGradient, deviceHysOutput, matrixMemorySize, cudaMemcpyDeviceToHost);

	cudaFreeArray(deviceMatrixArray);
	cudaFree(deviceGradient);
	cudaFree(deviceEdgeDirections);
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

void ParseArguments(int argc, char** argv)
{
	for (int i =0; i<argc; ++i)
	{
		if(strcmp(argv[i],"--file") == 0 || strcmp(argv[i],"-file") == 0) {
			BMPInFile = argv[i+1];
			i = i+1;
		}

		if(strcmp(argv[i],"--out") == 0 || strcmp(argv[i],"-out") == 0) {
			BMPOutFile = argv[i+1];
			i = i+1;
		}
	}
}


int main(int argc, char** argv)
{
 	ParseArguments(argc, argv);
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
	unsigned int *edgeDirectionClassifications = NULL;
	float time_smooth = 0;
	float time_gradient = 0;
	float time_suppression = 0;
	float time_hysteresis = 0;

	BitMapRead(BMPInFile, &bmp, &dib, &data, &palete);
	out = (unsigned char *)malloc(dib.image_size);
	edgeDirectionClassifications = (unsigned int *)malloc(dib.image_size);
	cudaEventRecord(start);
	//Gaussian Smoothening
        auto begin2 = chrono::high_resolution_clock::now();                            
	computeGradient(data, dib.width, out,edgeDirectionClassifications, &time_smooth, &time_gradient, &time_suppression, &time_hysteresis);
	//convolution(data, out, dib.width, gaussianMask, 5, gaussianMaskWeight);
	BitMapWrite("GPU_Gaussian_Smooth_Gradient_suppression2.bmp", &bmp, &dib, out, palete);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	data = NULL;
	free(data);
	//suppressNonmaximums(dib.width, edgeDirectionClassifications, out, dib.image_size);
	//BitMapWrite("GPU_Gaussian_Smooth_Gradient_suppression.bmp", &bmp, &dib, out, palete);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Input Image size %d x %d pixels \n", dib.height, dib.width); 
	printf("Elapsed Time for smoothing:%f ms\n",time_smooth);
	printf("Elapsed Time for gradient:%f ms\n",time_gradient);
	printf("Elapsed Time for suppression:%f ms\n",time_suppression);
	printf("Elapsed Time for hysteresis:%f ms\n",time_hysteresis);
	printf("Elapsed Time for total edge detection:%f ms\n",milliseconds);
	

}


			

			

