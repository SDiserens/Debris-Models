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
#include "Utilities\OrbitalUtilities.h"
#include "Utilities\OrbitalAnomalies.h"
#include "Utilities\OrbitalElements.h"
#include "Utilities\DebrisObject.h"
#include "Utilities\DebrisPopulation.h"
#include "Utilities\BaseUtilities.h"
