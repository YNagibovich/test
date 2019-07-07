// dummy image 
#ifndef _DUMMY_IMAGE_GPU_H_
#define _DUMMY_IMAGE_GPU_H_

#include <stdio.h>

bool isCUDA_OK();

bool cudaGrayscale( const unsigned char* pImageSrc, int nWidth, int nHeight);

bool cudaBlur( unsigned char* pImageDst, int nWidth, int nHeight, const float* pMatrix, int nParam);

bool cudaClose();

#endif // _DUMMY_IMAGE_GPU_H_



