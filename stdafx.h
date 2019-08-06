// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include "targetver.h"

#include <stdio.h>
#include <map>
#include <tchar.h>
#include <math.h>
#include <stdexcept>
#include <random>
#include <algorithm>
#include <map>

#include <iostream>
#include <iomanip>  
#include <fstream>
#include <sstream>
#include <string>

#include <json\json.h>
#include <ctime>
#include <chrono>

#include <memory>
using namespace std;



// reference additional headers your program requires here
#include "Utilities\OrbitalUtilities.h"
#include "Utilities\OrbitalAnomalies.h"
#include "Utilities\OrbitalElements.h"
#include "Utilities\DebrisObject.h"
#include "Utilities\DebrisPopulation.h"
#include "Utilities\BaseUtilities.h"

#include "Modules\Modules.h"
#include "Utilities\ModuleFactory.h"