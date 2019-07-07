// dummy image 
#include <iostream>
#include <thread>
#include <vector>
#include <algorithm>
#include <cmath>
#include <fstream>
#include "image.h"
#include "image_gpu.h"

#include "turbojpeg.h"

CImage::CImage( int nTotal) : m_pRawData(nullptr), m_pProcessedData(nullptr),
m_pAuxData(nullptr),m_nWidth(0), m_nHeight(0), m_bGPU(false)
{
    m_nTotalThreads = nTotal;

    m_bGPU = isGPU_OK();
}

CImage::~CImage()
{
    if (m_pRawData != nullptr)
    {
        delete [] m_pRawData;
    }

    if( m_pProcessedData != nullptr)
    {
        delete [] m_pProcessedData;
    }

    if( m_pAuxData != nullptr)
    {
        delete [] m_pAuxData;
    }

    if( m_bGPU)
    {
        cudaClose();
    }
}

bool CImage::isGPU_OK()
{
    if( !m_bGPU)
    {
        m_bGPU = isCUDA_OK();
    }

    return m_bGPU;
}

// jpeg only 
bool CImage::load( const char* pName)
{
    bool bRet = false;

    // TBD ref
    if( m_pRawData!=nullptr)
    {
        return bRet;
    }

    std::ifstream stream( pName, std::ifstream::binary);
    if(!stream)
    {
        return  bRet;
    }

    if (!stream.good())
    {
        stream.close();
        return bRet;
    }

    stream.seekg(0, std::ios::end);
    int nInputLength = (int)stream.tellg();
    stream.seekg(0, std::ios::beg);
    unsigned char* pJpegData = nullptr;
    int nJpegSize = nInputLength;
    try
    {
        pJpegData = new unsigned char[nJpegSize]; 
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return bRet;
    }
    
    stream.read(reinterpret_cast<char *>(pJpegData), nInputLength);
    stream.close();

    int jpegSubsamp = 0, width = 0, height = 0;
	tjhandle jpegDecompressor = tjInitDecompress();

    if (jpegDecompressor != nullptr)
	{
        if (tjDecompressHeader2(jpegDecompressor, pJpegData, nJpegSize, &width, &height, &jpegSubsamp) == 0)
        {
            int nLinesize = width * 4; // RGBA
			m_nWidth = width;
			m_nHeight = height;
			int nSize = nLinesize * height;
            bRet = true;
            try
            {
                m_pRawData = new unsigned char[nSize]; 
            }
            catch(const std::exception& e)
            {
                std::cerr << e.what() << '\n';
                bRet = false;
            }
			
			if (bRet)
			{
                if (tjDecompress2(jpegDecompressor, pJpegData, nJpegSize,
					m_pRawData, width, nLinesize, height, TJPF_RGBX, TJFLAG_FASTDCT) == 0)
				{
					// debug
				}
				else
				{
                    bRet = false;
				}
			}
		}
		tjDestroy(jpegDecompressor);
    }
    delete [] pJpegData;
    return bRet;
}
    
// jpeg only 
bool CImage::save( const char* pName)
{
    bool bRet = false;
    long unsigned int nCompressedSize = 0;
    unsigned char* pCompressedData = nullptr;
	int	nSize = m_nWidth * m_nHeight;

    tjhandle        jpegEncoder = nullptr;
    int             flags = TJFLAG_ACCURATEDCT;

    jpegEncoder = tjInitCompress();
    if (jpegEncoder == nullptr)
    {
        return bRet;
    }

    try
    {
        pCompressedData = new unsigned char[nSize]; 
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        tjDestroy(jpegEncoder);  

        return bRet;
    }
    
    bRet = true;
    if( tjCompress2(jpegEncoder, m_pProcessedData,  m_nWidth, m_nWidth,
            m_nHeight, TJPF_GRAY, &pCompressedData, &nCompressedSize, TJSAMP_GRAY, DEF_QUALITY, flags) < 0)
    {
        bRet = false;
    }

    tjDestroy(jpegEncoder);  

    if( bRet)
    {
        bRet = false;
        std::ofstream stream( pName, std::ofstream::binary);
        if (!stream.good())
        {
            bRet = false;
        }
        stream.write( (const char*)pCompressedData, nCompressedSize);
        bRet = true;
    }

    if( pCompressedData!=nullptr)
    {
        delete [] pCompressedData;
    }

    return bRet;
}

unsigned char* CImage::getInputOffset( int nOrder)
{
    // TBD ref
    int nSliceSize = m_nWidth * m_nHeight / m_nTotalThreads * 4;

    unsigned char* pRet = m_pRawData + nSliceSize * nOrder;

    return pRet;
}

unsigned char* CImage::getOutputOffset( int nOrder)
{
    // TBD ref
    int nSliceSize = m_nWidth * m_nHeight / m_nTotalThreads;

    unsigned char* pRet = m_pProcessedData + nSliceSize * nOrder;

    return pRet;
}

unsigned char* CImage::getAuxOffset( int nOrder)
{
    // TBD ref
    int nSliceSize = m_nWidth * m_nHeight / m_nTotalThreads;

    unsigned char* pRet = m_pAuxData + nSliceSize * nOrder;

    return pRet;
}

bool CImage::toGrayscale( int nThreads)
{
    bool bRet = false;

    if( m_pProcessedData == nullptr)
    {
        try
        {
            m_pProcessedData = new unsigned char[m_nWidth * m_nHeight]; 
        }
        catch(const std::exception& e)
        {
                std::cerr << e.what() << '\n';
                return bRet;
        }
    }

    if( is_GPU_OK())
    {
        bRet = cudaGrayscale( m_pRawData, m_nWidth, m_nHeight);
    }
    else
    {
        bRet = true;
        std::vector<std::thread> vWorkers;

        for (int i=0; i<nThreads; i++)
        {
            std::thread tWorker;
                
            tWorker = std::thread(&CImage::_toGrayscale, this, i);
            vWorkers.push_back(std::move(tWorker));
        }
        
        for ( std::thread& it : vWorkers)
        {
            if( it.joinable())
            {
                it.join();
            }
        }
    }
    return bRet;
}


// rgba->gray
bool CImage::_toGrayscale( int nOrder)
{
    bool bRet = true;

    //uint32_t* pSrc = (uint32_t*)m_pRawData;
    uint32_t* pSrc = (uint32_t*)getInputOffset( nOrder);

    //uint8_t* pDst = (uint8_t*)m_pProcessedData;
    uint8_t* pDst = (uint8_t*)getOutputOffset( nOrder);
    
    // assume stride = width
    
    int nSliceHeight = getSliceSize();

    for( int y = 0; y< nSliceHeight; y++)
    {
        for( int x = 0; x<m_nWidth; x++)
        {
            uint8_t* ptr = (uint8_t*)pSrc;
            float dR = (float)*ptr++;
            float dG = (float)*ptr++;;
            float dB = (float)*ptr;
            float dVal = 0.299 * dR + 0.587 * dG + 0.114 * dB;

            *pDst++ =( dVal > 255 ? 255 : (uint8_t) dVal);
            *pSrc++;
        }
    }

    return bRet;
}    

bool CImage::blur( int nThreads)
{
    bool bRet = false;

    if( is_GPU_OK())
    {
        bRet = cudaBlur( m_pProcessedData, m_nWidth, m_nHeight);
    }
    else
    {
        try
        {
            m_pAuxData = new unsigned char[m_nWidth * m_nHeight]; 
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            return bRet;
        }

        bRet = true;
        std::vector<std::thread> vWorkers;

        for (int i=0; i<nThreads; i++)
        {
            std::thread tWorker;
                
            tWorker = std::thread(&CImage::_blur, this, i);
            vWorkers.push_back(std::move(tWorker));
        }

        for ( std::thread& it : vWorkers)
        {
            if( it.joinable())
            {
                it.join();
            }
        }
    }
    return bRet;
}

// Gaussian blur, deep 3
bool CImage::_blur( int nOrder)
{
    bool bRet = true;
    float fParam = 3;
	float sigma2=fParam * fParam;
	int size=5; 
	float pixel = 0;
	float sum = 0;
	
    // TBD use member
    int nSliceHeight = getSliceSize();
    uint8_t* pDst = (uint8_t*)getOutputOffset( nOrder);
    uint8_t* pAux = (uint8_t*)getAuxOffset( nOrder);

	//blur x components
	for(int y=0; y < nSliceHeight; y++)
	{
		for(int x=0; x < m_nWidth; x++)
		{
			sum=0;
			pixel=0;

			//calc
			for(int i = std::max( 0, x - size); i <= std::min( m_nWidth - 1, x + size); i++)
			{
				float factor = std::exp(-(i - x)*(i - x)/(2 * sigma2));
				sum += factor;
                pixel += (factor * pDst[(i + y * m_nWidth)]);
			};
            pAux[(x + y * m_nWidth)] = pixel/sum;
		};
	};

	//blur y components
	for(int y=0; y<nSliceHeight; y++)
	{
		for(int x=0; x<m_nWidth; x++)
		{
			sum=0;
			pixel=0;

			//calc
			for(int i = std::max( 0, y - size); i <= std::min( nSliceHeight - 1, y + size); i++)
			{
				float factor = std::exp(-(i - y)*(i - y)/(2 * sigma2));
				sum += factor;
                pixel += (factor * pAux[( x + i * m_nWidth)]);
			};
            pDst[ ( x + y * m_nWidth)] = pixel/sum;
		};
	};
    return bRet;
}

