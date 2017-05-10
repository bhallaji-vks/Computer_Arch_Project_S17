//Image read, Smoothing operation and gradient operation performed
// Bhallaji Venkatesan and Divya Sampath Kumar

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
//#include <sys/io.h>
//#include <sys/types.h>
//#include <sys/stat.h>
//#include <cutil_inline.h>

#include "bmp.h"

char *BMPInFile = "lena512.bmp";
char *BMPOutFile = "gradient.bmp";


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
	BitMapRead(BMPInFile, &bmp, &dib, &data, &palete);
	out = (unsigned char *)malloc(dib.image_size);
	xGradient = (unsigned char *)malloc(dib.image_size);
	yGradient = (unsigned char *)malloc(dib.image_size);
	gradient = (unsigned char *)malloc(dib.image_size);
	convolution(data, out, dib.width, gaussianMask, 5, gaussianMaskWeight);
	convolution(out, xGradient, dib.width, xGradientMask, 3, 4);
	convolution(out, yGradient, dib.width, yGradientMask, 3, 4);
	for(unsigned i = 0; i < dib.width; ++i)
	{
		for(unsigned j = 0; j < dib.width; ++j)
		{
			unsigned matrixIndex = i * dib.width + j;
			gradient[matrixIndex] = fabs(xGradient[matrixIndex]) + fabs(yGradient[matrixIndex]);
		}
	}
	//CPU_Boost(data, out, dib.width, dib.height);

	//BitMapWrite(BMPOutFile, &bmp, &dib, out, palete);
	BitMapWrite(BMPOutFile, &bmp, &dib, gradient, palete);
}
	

	
