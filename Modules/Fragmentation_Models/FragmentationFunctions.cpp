// FragmentationFucntions.cpp : contains the implementation of different functions specific to fragmentation models.
//
#include "stdafx.h"
#include "fragmentation.h"

void MergeFragmentPopulations(DebrisPopulation population, FragmentCloud cloud)
{

}

float CalculateEnergyToMass(float kineticEnergy, float mass) // Returns E/m ratio in J/g
{
	float energyToMass = kineticEnergy / (1000 * mass);
	return energyToMass;
}