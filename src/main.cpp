#include <iostream>
#include <stdio.h>
#include <thread>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>

#include "image.h"

//////////////////////////////////////////////////////////////////////////
// defaults

#define DEF_OUTPUT_NAME "processed.jpg"


//////////////////////////////////////////////////////////////////////////
// info

void print_usage()
{
    std::cout << "usage : program path_to_image.jpg [N] " << std::endl;
    std::cout << "\tN               - number of threads" << std::endl;
}

//////////////////////////////////////////////////////////////////////////
// utils

bool check_params(int argc, char* argv[])
{
    if (argc < 2)
    {
        print_usage();
        return false;
    }
    return true;
}

uint64_t getTimeMilliseconds(void)
{
    struct timeval tp;
    gettimeofday(&tp, 0);
    return ((uint64_t)tp.tv_sec) * 1000 + tp.tv_usec / 1000;
}

//////////////////////////////////////////////////////////////////////////
// the main

int main(int argc, char* argv[])
{
    if (!check_params( argc, argv))
    {
        return -1;
    }

    bool bDone = false;

    uint64_t tStart = getTimeMilliseconds();
    uint64_t tStage1 = 0;
    uint64_t tStage2 = 0;
    uint64_t tStage3 = 0;
    uint64_t tStage4 = 0;

    int nThreads = std::thread::hardware_concurrency();

    if( argc >2)
    {
        nThreads = atoi(argv[2]);
    }

    if( nThreads <1)
    {
        nThreads = 1;
    }

    std::cout << "Threads : " << nThreads << std::endl;

    CImage image(nThreads);

    if( !image.load(argv[1]))
    {
        std::cout << "Failed to load " << argv[1] << std::endl;
    }
    else
    {
    
        tStage1 = getTimeMilliseconds();
        image.toGrayscale( nThreads);
        tStage2 = getTimeMilliseconds();
        image.blur( nThreads);
        tStage3 = getTimeMilliseconds();
        bDone = image.save( DEF_OUTPUT_NAME);
        tStage4 = getTimeMilliseconds();
    }
    // print stat    
    if( bDone)
    {
        std::cout << "Load in "<< tStage1 - tStart << "ms."<< std::endl;
        std::cout << "toGrayscale in "<< tStage2 - tStage1 << "ms."<< std::endl;
        std::cout << "Blur in "<< tStage3 - tStage2 << "ms."<< std::endl;
        std::cout << "Save in "<< tStage4 - tStage3 << "ms."<< std::endl;
        std::cout << "Total time spent : "<< tStage4 - tStart << "ms."<< std::endl;
    }
    else
    {
        std::cout << "Something went wrong!"<< std::endl;
    }
    
    return bDone ? 0 : -1;
}

