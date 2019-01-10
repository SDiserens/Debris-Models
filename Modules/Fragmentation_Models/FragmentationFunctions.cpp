// FragmentationFucntions.cpp : contains the implementation of different functions specific to fragmentation models.
//
#include "stdafx.h"
#include "fragmentation.h"

double  catastrophicThreshold = 40;

void MergeFragmentPopulations(DebrisPopulation& currentPopulation, FragmentCloud& cloud)
{
	// Add event MetaData
	Event tempEvent(currentPopulation.GetEpoch(),
					cloud.explosion,
					cloud.consMomentumFlag, 
					(cloud.energyMassRatio > catastrophicThreshold),
					cloud.totalMass,
					cloud.debrisCount);

	currentPopulation.eventLog.push_back(tempEvent);

	// Merge population
	for (auto &bucketCloud : cloud.fragmentBuckets)
	{
		for(auto & debris : bucketCloud.fragments)
		{
			currentPopulation.population.push_back(debris);
		}
	}
}

double CalculateEnergyToMass(double kineticEnergy, double mass) // Returns E/m ratio in J/g
{
	double energyToMass = kineticEnergy / (1000 * mass);
	return energyToMass;
}