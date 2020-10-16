// Includes generic functionality for use by all fragmentation models
//
#pragma once

// Reference to various headers required by fragmentation models
#include "FragmentCloud.h"

// Declaration of generic fragmentation functions - definition in FragmentationFunctions.cpp

void MergeFragmentPopulations(DebrisPopulation& population, FragmentCloud& cloud, Event& fragmentationEvent, double massLimit);

double CalculateEnergyToMass(double kineticEnergy, double mass);

class BreakupModel
{
public:
	double minLength, representativeFragmentThreshold, massLimit, catastrophicThreshold;// J/g of target mass
	int representativeFragmentNumber;

	virtual void mainBreakup(DebrisPopulation& population, Event& fragmentationEvent) = 0;
	void SetMassLimit(double mass);
	
};