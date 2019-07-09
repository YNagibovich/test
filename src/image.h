// dummy image 
#ifndef _DUMMY_IMAGE_H_
#define _DUMMY_IMAGE_H_

#ifndef NOCUDA

#include "image_gpu.h"

#endif


class CImage 
{
public:
    CImage( int nTotal);
    ~CImage();

    // jpeg only 
    bool load( const char* pName);
    
    // jpeg only 
    bool save( const char* pName);

    // rgba->gray
    bool toGrayscale( int nThreads = 1);

    // Gaussian blur, deep 3
    bool blur( int nThreads = 1);

private:

    bool isGPU_OK();
    bool createMatrix( int nParam);

    // rgba->gray
    bool _toGrayscale( int nOrder);

    // Gaussian blur, deep 3
    bool _blur( int nOrderint);

    unsigned char* getInputOffset( int nOrder);
    unsigned char* getOutputOffset( int nOrder);
    unsigned char* getAuxOffset( int nOrder);
    int getSliceSize() { return m_nHeight / m_nTotalThreads;}

    unsigned char* m_pRawData;
    unsigned char* m_pProcessedData;
    unsigned char* m_pAuxData;;
    float* m_pMatrix;         
    int m_nWidth;
    int m_nHeight;
    int m_nTotalThreads;
    bool m_bGPU;
};

#define DEF_QUALITY 100

#endif // _DUMMY_IMAGE_H_