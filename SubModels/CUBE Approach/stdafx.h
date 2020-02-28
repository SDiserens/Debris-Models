// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include "targetver.h"

#include <stdio.h>
#include <tchar.h>
#include <math.h>
#include <stdexcept>
#include <random>
#include <map>
#include <algorithm>
#include <unordered_map> 

#include <iostream>
#include <iomanip>  
#include <fstream>
#include <sstream>
#include <string>
#include <ctime>
#include <chrono>
#include <list>
#include <json\json.h>

using namespace std;
#include "ppl.h"
#include <mutex>

// reference additional headers your program requires here
#include "Utilities/OrbitalUtilities.h"
#include "Utilities/OrbitalAnomalies.h"
#include "Utilities/OrbitalElements.h"
#include "Utilities/DebrisObject.h"
#include "Utilities/DebrisPopulation.h"
#include "Utilities/BaseUtilities.h"