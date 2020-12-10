// FragmentationFucntions.cpp : contains the implementation of different functions specific to fragmentation models.
//
#include "stdafx.h"
#include "fragmentation.h"

double  catastrophicThreshold = 40;

void MergeFragmentPopulations(DebrisPopulation& currentPopulation, FragmentCloud& cloud, Event& fragmentationEvent, double massLimit)
{
	// Prevent accidental breakup of collision avoidance event
	if (fragmentationEvent.GetEventType() >= 2) return;
	int debrisCount = 0;
	// Add event MetaData
	bool isIntact = currentPopulation.GetObject(fragmentationEvent.GetPrimary()).IsIntact();
	fragmentationEvent.SetCatastrophic(isIntact * (cloud.energyMassRatio > catastrophicThreshold));
	fragmentationEvent.SetConservationMomentum(cloud.consMomentumFlag);
	fragmentationEvent.SetEMR(cloud.energyMassRatio);

	currentPopulation.RemoveObject(fragmentationEvent.GetPrimary(), fragmentationEvent.GetEventType());

	// Merge population
	for (auto &bucketCloud : cloud.fragmentBuckets)
	{
		for(auto & debris : bucketCloud.fragments)
		{
			if ((debris.GetElements().eccentricity < 1) && (debris.GetMass() > massLimit)) {
				currentPopulation.AddDebrisObject(debris);
				++debrisCount;
			}
			else
				debris.~DebrisObject();
		}
	}

	fragmentationEvent.SetDebrisCount(debrisCount);
	currentPopulation.AddDebrisEvent(fragmentationEvent);
	cloud.ClearCloud();
	//cloud.~FragmentCloud();
}

double CalculateEnergyToMass(double kineticEnergy, double mass) // Returns E/m ratio in J/g
{
	double energyToMass = kineticEnergy / (1000 * mass);
	return energyToMass;
}