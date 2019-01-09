// Includes generic functionality for use by all fragmentation models
//
#pragma once

// Reference to various headers required by fragmentation models
#include "FragmentCloud.h"

// Declaration of generic fragmentation functions - definition in FragmentationFunctions.cpp

void MergeFragmentPopulations(DebrisPopulation population, FragmentCloud cloud);

double CalculateEnergyToMass(double kineticEnergy, double mass);
extern double catastrophicThreshold; // J/g of target mass