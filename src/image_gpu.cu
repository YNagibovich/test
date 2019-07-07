// dummy image 
#include "image_gpu.h"
#include <cuda_runtime.h>


// TBD REF
static void* g_imgSrc = nullptr;
static void* g_imgDst = nullptr;
static void* g_imgAux = nullptr;


// TBD REF, use context
bool isCUDA_OK()
{
    int nCnt = 0;
    bool bRet = false;

    cudaError_t error_id = cudaGetDeviceCount(&nCnt);
    if (error_id == cudaSuccess)
    {
        bRet = (nCnt > 0);
    }
    return bRet;
}

bool cudaClose()
{
    if( g_imgSrc)    
    {
        cudaFree(g_imgSrc);
    }
    if( g_imgDst)    
    {
        cudaFree(g_imgDst);
    }
    if( g_imgAux)    
    {
        cudaFree(g_imgAux);
    }
}

bool cudaGrayscale( unsigned char* pImageSrc, int nWidth, int nHeight)
{
    bool bRet = false;
    
    if( !pImageSrc)
    {
        return bRet;
    }

    // alloc buffers
    if( !g_imgSrc)
    {
        cudaError_t error_id = cudaMalloc( (void **)&g_imgSrc, nWidth * nHeight * 4 ); // assume RGBA
    }
    if( !g_imgSrc)
    {
        return bRet;
    }

    if( !g_imgDst)
    {
        cudaError_t error_id = cudaMalloc( (void **)&g_imgDst, nWidth * nHeight); // assume GS
    }
    if( !g_imgDst)
    {
        return bRet;
    }

    if( !g_imgAux)
    {
        cudaError_t error_id = cudaMalloc( (void **)&g_imgAux, nWidth * nHeight); // assume GS
    }
    if( !g_imgAux)
    {
        return bRet;
    }

    // load data
    cudaError_t error_id = cudaMemcpy( g_imgSrc, pImageSrc, nWidth * nHeight * 4, cudaMemcpyHostToDevice );

    // start processing

    bRet = true;

    return bRet;
}

bool cudaBlur( unsigned char* pImageDst, int nWidth, int nHeight)
{
    bool bRet = false;

    if( !pImageDst)
    {
        return bRet;
    }

    if( !g_imgDst || !g_imgAux)
    {
        return bRet;
    }

    // start processing
    
    bRet = true;
    // save data
    cudaError_t error_id = cudaMemcpy( (void*)pImageDst, g_imgDst, nWidth * nHeight, cudaMemcpyDeviceToHost );

    return bRet;
}







