// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

#ifdef __CUDACC__
#define CUDA_CLASS thrust::
#else
#define CUDA_CLASS std::
#endif 

#ifdef __CUDACC__
#define CUDA_HOST_MEMBER __host__
#else
#define CUDA_HOST_MEMBER
#endif 

#ifdef __CUDACC__
#define CUDA_CONSTANT __constant__
#else
#define CUDA_CONSTANT
#endif 

#include "targetver.h"

#include <stdio.h>
#include <map>
#include <tchar.h>
#include <math.h>
#include <stdexcept>
#include <random>
#include <algorithm>
#include <map>

#include <unordered_map> 

#include <iostream>
#include <iomanip>  
#include <fstream>
#include <sstream>
#include <string>

#include <list>
#include <json\json.h>
#include <ctime>
#include <chrono>

#include <memory>
using namespace std;

#include "ppl.h"
#include <mutex>


// reference additional headers your program requires here
#include "Utilities\OrbitalUtilities.cuh"
#include "Utilities\OrbitalAnomalies.cuh"
#include "Utilities\OrbitalElements.cuh"
#include "Utilities\DebrisObject.cuh"
#include "Utilities\DebrisEvent.cuh"
#include "Utilities\DebrisPopulation.h"
#include "Utilities\BaseUtilities.h"
