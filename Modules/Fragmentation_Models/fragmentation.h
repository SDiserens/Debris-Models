// Includes generic functionality for use by all fragmentation models
//
#pragma once

// Reference to various headers required by fragmentation models
#include "../../Framework/stdafx.h"
#include "FragmentCloud.h"

// Declaration of generic fragmentation functions - definition in FragmentationFunctions.cpp

void MergeFragmentPopulations(DebrisPopulation population, FragmentCloud cloud);