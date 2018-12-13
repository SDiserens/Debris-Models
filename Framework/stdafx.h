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
using namespace std;

const double NEWTONMAXITERATIONS = 20;
const double NEWTONTOLERANCE = 1e-13;

const double Pi = _Pi;
const double Tau = 2 * _Pi;


// TODO: reference additional headers your program requires here
#include "Utilities\OrbitalUtilities.h"
#include "Utilities\OrbitalElements.h"
#include "Utilities\OrbitalAnomalies.h"
#include "Utilities\DebrisObject.h"
#include "Utilities\DebrisPopulation.h"