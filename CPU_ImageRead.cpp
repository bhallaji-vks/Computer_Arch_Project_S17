//Image read, Smoothing operation performed
// Bhallaji Venkatesan and Divya Sampath Kumar

#define _DEFINE_DEPRECATED_HASH_CLASSES 0
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
/*class EdgeDetection
{	void visitNeighbors(int i, int j, float lowThreshold, unsigned char* gradientImage, unsigned char* outputEdges, int width);
	void performHysteresis(unsigned char* gradientImage, float highThreshold, float lowThreshold, unsigned char* outputEdges, int width) 
	std::hash_set<unsigned> visitedPixels;
};
*/

using namespace std;

//using namespace stdext;
//hash_set<unsigned> visitedPixels;
std::unordered_set<unsigned int> visitedPixels;

//#include <sys/io.h>
//#include <sys/types.h>
//#include <sys/stat.h>
//#include <cutil_inline.h>

#include "bmp.h"

char *BMPInFile = "meena.bmp";
char *BMPOutFile = "suppress.bmp";

//vector<int> visited_vec;‘visitedPixels’
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

void CPU_Boost(unsigned char* image_in, unsigned char* image_out, int width, int height)
{
	int i, j, rows, col;
	float a =224.0;
	rows = height;
	col = width;
	for(i =0;i<rows;i++)
	{
		for(j=0;j<col; j++)
		{
			image_out[i*width +j] = image_in[i*width +j];
		}
	}
}

const float gaussianMask[25] =
{
	2,  4,  5,  4, 2,
	4,  9, 12,  9, 4,
	5, 12, 15, 12, 5,
	4,  9, 12,  9, 4,
	2,  4,  5,  4, 2
};

const float xGradientMask[9] = 
{
	-1, 0, 1,
	-2, 0, 2,
	-1, 0, 1
};

const float yGradientMask[9] = 
{
	 1,  2,  1,
	 0,  0,  0,
	-1, -2, -1
};


const float gaussianMaskWeight = 159;


void convolution(const unsigned char* image_in, unsigned char* image_out, int width, const float* mask, int mask_width, float mask_weight)
{
	int mask_offset = (mask_width -1)/2;
	for(int outputRow = 0; outputRow < width; ++outputRow)
	{
		for(int outputColumn = 0; outputColumn < width; ++outputColumn)
		{
			
			float accumulator = 0;
			for(int maskRow = -mask_offset; maskRow <= mask_offset; ++maskRow)
			{
				
				for(int maskColumn = -mask_offset; maskColumn <= mask_offset; ++maskColumn)
				{
					
					int maskIndex = (mask_offset + maskRow) * mask_width + (mask_offset + maskColumn);
					int matrixRow = outputRow - (mask_offset - 2 - maskRow);
					int matrixColumn = outputColumn - (mask_offset - 2 - maskColumn);
					int matrixIndex = matrixRow * width + matrixColumn;

					if(matrixRow >= 0 && matrixColumn >= 0 && matrixRow < width && matrixColumn < width)
					{
						accumulator += mask[maskIndex] * image_in[matrixIndex];
					}
				}
			}
			image_out[outputRow * width + outputColumn] = accumulator / mask_weight;
		}
	} 

}


int getClockwisePerpendicularIndex(unsigned i, unsigned j, unsigned edgeDirectionClassification, int width) 
{
	int clockwisePerpendicularI;
	int clockwisePerpendicularJ; 
	switch(edgeDirectionClassification)
	{
		case 0:
			clockwisePerpendicularI = i - 1;
			clockwisePerpendicularJ = j;
			break;
		case 1:
			clockwisePerpendicularI = i - 1;
			clockwisePerpendicularJ = j + 1;
			break;
		case 2:
			clockwisePerpendicularI = i;
			clockwisePerpendicularJ = j + 1;
			break;
		case 3:
			clockwisePerpendicularI = i + 1;
			clockwisePerpendicularJ = j + 1;
			break;
	}
	
	//cout << "\tClockwise perpendicular pixel: (" << clockwisePerpendicularI << ", " << clockwisePerpendicularJ << ") = ";

	if(clockwisePerpendicularI < 0 || clockwisePerpendicularJ < 0 || clockwisePerpendicularI >= width || clockwisePerpendicularJ >= width)
	{
		return -1;
	}
	else
	{
		return clockwisePerpendicularI * width + clockwisePerpendicularJ;
	}
}

int getCounterClockwisePerpendicularIndex(unsigned i, unsigned j, unsigned edgeDirectionClassification, int width) 
{
	int counterClockwisePerpendicularI;
	int counterClockwisePerpendicularJ; 
	switch(edgeDirectionClassification)
	{
		case 0:
			counterClockwisePerpendicularI = i + 1;
			counterClockwisePerpendicularJ = j;
			break;
		case 1:
			counterClockwisePerpendicularI = i + 1;
			counterClockwisePerpendicularJ = j - 1;
			break;
		case 2:
			counterClockwisePerpendicularI = i;
			counterClockwisePerpendicularJ = j - 1;
			break;
		case 3:
			counterClockwisePerpendicularI = i - 1;
			counterClockwisePerpendicularJ = j - 1;
			break;
	}

	//cout << "\tCounterclockwise perpendicular pixel: (" << counterClockwisePerpendicularI << ", " << counterClockwisePerpendicularJ << ") = ";

	if(counterClockwisePerpendicularI < 0 || counterClockwisePerpendicularJ < 0 ||counterClockwisePerpendicularJ >= width || counterClockwisePerpendicularJ >= width)
	{
		return -1;
	}
	else
	{
		return counterClockwisePerpendicularI * width + counterClockwisePerpendicularJ;
	}
}


void classifyEdgeDirections(int imgsize, unsigned int* edgeDirectionClassifications, unsigned char* edgeDirections)
//void classifyEdgeDirections(int imgsize, unsigned int* edgeDirectionClassifications)
{
	int c0,c1,c2,c3;	
	for(unsigned int i = 0; i < imgsize; ++i)
	{
		//unsigned char edgeDirection = 10;
		unsigned char edgeDirection = edgeDirections[i];
		if((edgeDirection >= (unsigned char)0.0 && edgeDirection < (unsigned char)22.5) ||(edgeDirection >= (unsigned char)157.5 && edgeDirection < (unsigned char)202.5) || (edgeDirection >= (unsigned char)337.5 && edgeDirection <= (unsigned char)360.0))
		{
			edgeDirectionClassifications[i] = 0;c0++;
		}
		else if((edgeDirection >= (unsigned char)22.5 && edgeDirection < (unsigned char)67.5) || (edgeDirection >= (unsigned char)202.5 && edgeDirection < (unsigned char)247.5))
		{
			edgeDirectionClassifications[i] = 1;c1++;
		}
		else if((edgeDirection >= (unsigned char)67.5 && edgeDirection < (unsigned char)112.5) || (edgeDirection >= (unsigned char)247.5 && edgeDirection < (unsigned char)292.5))
		{
			edgeDirectionClassifications[i] = 2;c2++;
		}
		else if((edgeDirection >= (unsigned char)112.5 && edgeDirection < (unsigned char)157.5) || (edgeDirection >= (unsigned char)292.5 && edgeDirection < (unsigned char)337.5))
		{
			edgeDirectionClassifications[i] = 3;c3++;
		}
	} 
}

void suppressNonmaximums(int width, unsigned int* edgeDirectionClassifications, unsigned char* gradient, int imgsize)
{

	for(unsigned int i = 0; i < width; ++i)
	{
		for(unsigned int j = 0; j < width; ++j)
		{
			unsigned int pixelIndex = i * width + j;
			int clockwisePerpendicularIndex = getClockwisePerpendicularIndex(i, j, edgeDirectionClassifications[pixelIndex], width);	
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
			int counterClockwisePerpendicularIndex = getCounterClockwisePerpendicularIndex(i, j, edgeDirectionClassifications[pixelIndex], width);
			
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
			
			if(gradient[pixelIndex] <= clockwisePerpendicularValue || gradient[pixelIndex] <= counterClockwisePerpendicularValue)
			{
				//cout << "\tPixel suppressed." << endl;
				gradient[pixelIndex] = 0;
				
			}
			else
			{
				//cout << "\tPixel retained." << endl;

			}
		}
	}
}

void computeEdgeDirections(unsigned char* gradient, int imgsize, unsigned char* edgeDirections, unsigned char* xGradient, unsigned char* yGradient, unsigned int* edgeDirectionClassifications, int width)
{
	
	//edge direction
		
	unsigned char* xGradient_local = NULL;
	unsigned char* yGradient_local = NULL;
	unsigned int* edc_local = NULL;
	float *x = NULL;
	float *y =NULL;
	xGradient_local = (unsigned char *)malloc(imgsize);
	yGradient_local = (unsigned char *)malloc(imgsize);
	x = (float *)malloc(imgsize);
	edc_local = (unsigned int *)malloc(imgsize);
	y = (float *)malloc(imgsize);
	
	for(unsigned int i = 0; i < imgsize; ++i)
	{ 	
		*(xGradient_local+i) = *(xGradient+i);
		*(yGradient_local+i) = *(yGradient+i);
		*(x+i) = float(*(xGradient_local+i));	
		*(y+i) = float(*(yGradient_local+i));
		*(edc_local+i) = *(edgeDirectionClassifications + i);
	}

	
	for(unsigned int i = 0; i < imgsize; ++i)
	{ 	
		float x_l = *(x+i);
		float y_l = *(y+i);
		float l = ((atan2 (3.9, 1.0) * (180 / 3.14159265))+180);
		edgeDirections[i]= (unsigned char)((atan2 (x_l, y_l) * (180 / 3.14159265))+180);
		//atan2(float(*(xGradient_local+i)), float(*(yGradient_local+i))) * (180 / 3.14159265)+ 180.0
	}
	
	
	//edge classification
	//classifyEdgeDirections(imgsize, edgeDirectionClassifications, edgeDirections);
	classifyEdgeDirections(imgsize, edc_local, edgeDirections);
	
	//suppress non-maximum
	suppressNonmaximums(width, edc_local, gradient, imgsize);
	
}

//void insert(int pixelIndex, unsigned int* in_arr, int imgsize)
//{
//	for(unsigned int i=0;i<imgsize;i++)
//	{
//		in_arr[i] = pixelIndex;
//	}
//}

//void find(int pixelIndex)
//{
	


void visitNeighbors(int i, int j, float lowThreshold, unsigned char* gradientImage, unsigned char* outputEdges, int width, int imgsize)
{
	int pixelIndex = i * width + j;
        
	if(i == 0 || j == 0 || i == width - 1 || j == width - 1 || visitedPixels.find(pixelIndex) != visitedPixels.end()  ||gradientImage[pixelIndex] <  lowThreshold)
	{
		//(pixelIndex);
		visitedPixels.insert(pixelIndex);
		return;
	}

	outputEdges[pixelIndex] =STRONG_EDGE;
	visitedPixels.insert(pixelIndex);

	visitNeighbors(i - 1, j - 1, lowThreshold, gradientImage, outputEdges, width, imgsize);
	visitNeighbors(i - 1, j, lowThreshold, gradientImage, outputEdges, width, imgsize);
	visitNeighbors(i - 1, j + 1, lowThreshold, gradientImage, outputEdges, width, imgsize);
	visitNeighbors(i, j + 1, lowThreshold, gradientImage, outputEdges, width, imgsize);
	visitNeighbors(i + 1, j + 1, lowThreshold, gradientImage, outputEdges, width, imgsize);
	visitNeighbors(i + 1, j, lowThreshold, gradientImage, outputEdges, width, imgsize);
	visitNeighbors(i + 1, j - 1, lowThreshold, gradientImage, outputEdges, width, imgsize);
	visitNeighbors(i, j - 1, lowThreshold, gradientImage, outputEdges, width, imgsize);
} 


void performHysteresis(unsigned char* gradientImage, float highThreshold, float lowThreshold, unsigned char* outputEdges, int width, int imgsize) 
{
	unsigned int *in_arr = NULL;
	int c0,c1;	
	for(unsigned int i=0;i<imgsize;i++)
	{
		if(gradientImage[i] > highThreshold) c0++;
		else if(gradientImage[i] < lowThreshold)c1++;
		printf("%d\t",gradientImage[i]);
	}
	printf("%d\t%d\n",c0,c1);	
	for(int i = 0; i < width; ++i)
	{
		for(int j = 0; j < width; ++j)
		{
			unsigned pixelIndex = i * width + j;
			// Mark out borders and all pixels below the high threshold.
			if(gradientImage[pixelIndex] >= highThreshold)
			{
				visitedPixels.insert(pixelIndex);
				outputEdges[pixelIndex] = STRONG_EDGE;
				visitNeighbors(i, j, lowThreshold, gradientImage, outputEdges, width, imgsize);
				//insert(pixelIndex,in_arr, imgsize);
				//outputEdges[pixelIndex] = STRONG_EDGE;
			}
			/*
			else if(gradientImage[pixelIndex] < highThreshold && gradientImage[pixelIndex] > lowThreshold)
			{	
				visitNeighbors(i, j, lowThreshold, gradientImage, outputEdges, width, imgsize);
			}
			*/
			
		}
	}
}



int main()
{
	
	struct bmp_header bmp;
	struct dib_header dib;
	unsigned char *palete = NULL;
	unsigned char *data = NULL;
	unsigned char *out = NULL;
	unsigned char *xGradient = NULL;
	unsigned char *yGradient = NULL;
	unsigned char *gradient = NULL;
	unsigned char *edgeDirections2 = NULL;
	unsigned char *hys_output = NULL;
	float highThreshold = 100.0;
	float lowThreshold = 90.0;
	unsigned int *edgeDirectionClassifications = NULL;

	BitMapRead(BMPInFile, &bmp, &dib, &data, &palete);
	out = (unsigned char *)malloc(dib.image_size);
	xGradient = (unsigned char *)malloc(dib.image_size);
	yGradient = (unsigned char *)malloc(dib.image_size);
	gradient = (unsigned char *)malloc(dib.image_size);
	

	convolution(data, out, dib.width, gaussianMask, 5, gaussianMaskWeight);
	BitMapWrite("gaussian.bmp", &bmp, &dib, out, palete);
	//free(out);
	convolution(out, xGradient, dib.width, xGradientMask, 3, 1);
	convolution(out, yGradient, dib.width, yGradientMask, 3, 1);
	for(unsigned int i = 0; i < dib.width; ++i)
	{
		for(unsigned int j = 0; j < dib.width; ++j)
		{
			unsigned int matrixIndex = i * dib.width + j;
			gradient[matrixIndex] = fabs(xGradient[matrixIndex]) + fabs(yGradient[matrixIndex]);
		}
	}
	BitMapWrite("gradient_image2.bmp", &bmp, &dib, gradient, palete);
	out = NULL;
	free(out);
	data = NULL;
	free(data);
	//free(out);
	//free(data);
	edgeDirections2 = (unsigned char *)malloc(dib.image_size);
	hys_output = (unsigned char *)malloc(dib.image_size);
	edgeDirectionClassifications = (unsigned int *)malloc(dib.image_size);
//void computeEdgeDirections(unsigned char* gradient, int imgsize, float* edgeDirections, unsigned char* xGradient, unsigned char* yGradient, unsigned int* edgeDirectionClassifications, int width)
	//computeEdgeDirections(gradient,dib.image_size,edgeDirections2,xGradient,yGradient,edgeDirectionClassifications,dib.width);
	//BitMapWrite("gradient_image_edge_latest2.bmp", &bmp, &dib, gradient, palete);
	//CPU_Boost(data, out, dib.width, dib.height);

	//BitMapWrite("Sup_test.bmp", &bmp, &dib, gradient, palete);
	//BitMapWrite("suppress.bmp", &bmp, &dib, gradient, palete);

	performHysteresis(gradient,highThreshold,lowThreshold,hys_output,dib.width, dib.image_size);
	BitMapWrite("hysterisis_test_wosupp.bmp", &bmp, &dib, hys_output, palete);
}
	

	
