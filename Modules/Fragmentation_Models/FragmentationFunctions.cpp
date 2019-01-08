// FragmentationFucntions.cpp : contains the implementation of different functions specific to fragmentation models.
//
#include "stdafx.h"
#include "fragmentation.h"

void MergeFragmentPopulations(DebrisPopulation currentPopulation, FragmentCloud cloud)
{
	// ToDO - Add event MetaData
	Event tempEvent(currentPopulation.GetEpoch(),
					cloud.explosion,
					cloud.consMomentumFlag, 
					(cloud.energyMassRatio > catastrophicThreshold),
					cloud.totalMass,
					cloud.debrisCount);

	currentPopulation.AddDebrisEvent(tempEvent);

	// Merge population
	for (auto &bucketCloud : cloud.fragmentBuckets)
	{
		for(auto & debris : bucketCloud.fragments)
		{
			currentPopulation.AddDebrisObject(debris);
		}
	}
}

double CalculateEnergyToMass(double kineticEnergy, double mass) // Returns E/m ratio in J/g
{
	double energyToMass = kineticEnergy / (1000 * mass);
	return energyToMass;
}