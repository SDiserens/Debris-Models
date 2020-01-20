// FragmentationFucntions.cpp : contains the implementation of different functions specific to fragmentation models.
//
#include "stdafx.h"
#include "fragmentation.h"

double  catastrophicThreshold = 40;

void MergeFragmentPopulations(DebrisPopulation& currentPopulation, FragmentCloud& cloud, Event& fragmentationEvent)
{
	// Add event MetaData
	fragmentationEvent.SetCatastrophic(cloud.energyMassRatio > catastrophicThreshold);
	fragmentationEvent.SetConservationMomentum(cloud.consMomentumFlag);
	fragmentationEvent.SetEMR(cloud.energyMassRatio);
	fragmentationEvent.SetDebrisCount(cloud.debrisCount);

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