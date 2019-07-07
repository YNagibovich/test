// dummy image 
#include "image_gpu.h"
#include <cuda_runtime.h>

// TBD REF
static void* g_imgSrc = nullptr;
static void* g_imgDst = nullptr;
static void* g_imgAux = nullptr;
static void* g_fMatrix = nullptr;

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

    if( g_fMatrix)    
    {
        cudaFree(g_fMatrix);
    }

    return true;
}

__global__ void _k_rgba2gray( const uchar4* const pRGBA, unsigned char* const pGray, int nRows, int nColumns)
{
    int pos_x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int pos_y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if(pos_x >= nColumns || pos_y >= nRows)
    {
        return;
    }

    uchar4 rgba = pRGBA[pos_x + pos_y * nColumns];
    pGray[pos_x + pos_y * nColumns] = ( 0.299f * rgba.x + 0.587f * rgba.y + 0.114f * rgba.z); 
}

bool cudaGrayscale(const unsigned char* pImageSrc, int nWidth, int nHeight)
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
    const uchar4* pSrc = reinterpret_cast<const uchar4*>(g_imgSrc);
    unsigned char* pDst = reinterpret_cast<unsigned char*>(g_imgAux);
    
    const dim3 blockSize( 16, 16, 1);
    const dim3 gridSize( nWidth/blockSize.x+1, nHeight/blockSize.y+1, 1);

    _k_rgba2gray<<<gridSize,blockSize>>>( pSrc, pDst, nHeight, nWidth);
    cudaDeviceSynchronize(); // wait for completion

    // TBD check errors

    bRet = true;

    return bRet;
}

__global__ void _k_blur(const unsigned char* pSrc, unsigned char* pDst, int nRows, int nColumns, const float* const pMatrix, int nParam)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= nColumns || py >= nRows) 
    {
        return;
    }

    float c = 0.0f;

    for (int fx = 0; fx < nParam; fx++) 
    {
        for (int fy = 0; fy < nParam; fy++) 
        {
            int imagex = px + fx - nParam / 2;
            int imagey = py + fy - nParam / 2;
            imagex = min( max( imagex, 0), nColumns - 1);
            imagey = min( max( imagey, 0), nRows - 1);
            c += ( pMatrix[fy * nParam + fx] * pSrc[imagey * nColumns + imagex]);
        }
    }

    pDst[py*nColumns+px] = c;
}

bool cudaBlur( unsigned char* pImageDst, int nWidth, int nHeight, const float* pMatrix, int nParam)
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

    // prepare calc

    cudaError_t error_id = cudaMalloc( (void **)&g_fMatrix, nParam * nParam * sizeof(float)); 
    // TBD check errors    

    error_id = cudaMemcpy( g_fMatrix, pMatrix, sizeof(float) * nParam * nParam, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize(); // wait for completion

    // process data
    const dim3 blockSize( 16, 16, 1);
    const dim3 gridSize( nWidth/blockSize.x+1, nHeight/blockSize.y+1, 1);

    unsigned char* pDst = reinterpret_cast<unsigned char*>(g_imgDst);
    unsigned char* pSrc = reinterpret_cast<unsigned char*>(g_imgAux);
    float* pfMatrix = reinterpret_cast<float*>(g_fMatrix);

    _k_blur<<<gridSize,blockSize>>>( pSrc, pDst, nHeight, nWidth, pfMatrix, nParam);
    cudaDeviceSynchronize(); // wait for completion
    
    // TBD check errors
    bRet = true;

    // save data
    error_id = cudaMemcpy( (void*)pImageDst, g_imgDst, nWidth * nHeight, cudaMemcpyDeviceToHost );
    cudaDeviceSynchronize(); // wait for completion
    return bRet;
}







