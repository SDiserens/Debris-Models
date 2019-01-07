// FragmentationFucntions.cpp : contains the implementation of different functions specific to fragmentation models.
//
#include "stdafx.h"
#include "fragmentation.h"

void MergeFragmentPopulations(DebrisPopulation population, FragmentCloud cloud)
{
	if (cloud.consMomentumFlag)
	{

	}
}

double CalculateEnergyToMass(double kineticEnergy, double mass) // Returns E/m ratio in J/g
{
	double energyToMass = kineticEnergy / (1000 * mass);
	return energyToMass;
}