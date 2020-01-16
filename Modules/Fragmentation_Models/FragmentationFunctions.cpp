// FragmentationFucntions.cpp : contains the implementation of different functions specific to fragmentation models.
//
#include "stdafx.h"
#include "fragmentation.h"

double  catastrophicThreshold = 40;

void MergeFragmentPopulations(DebrisPopulation& currentPopulation, FragmentCloud& cloud)
{
	Event tempEvent;
	// Add event MetaData
	if (!cloud.explosion) {
		//TODO- Add logix to update exisiting event
		/*
		tempEvent = Event(currentPopulation.GetEpoch(),
							cloud.targetID, cloud.projectileID,
							cloud.relativeVelocity.vectorNorm(),
							cloud.consMomentumFlag,
							(cloud.energyMassRatio > catastrophicThreshold),
							cloud.totalMass,
							cloud.debrisCount);
							*/
	}
	else
		tempEvent = Event(currentPopulation.GetEpoch(), cloud.targetID,
						cloud.consMomentumFlag,
						(cloud.energyMassRatio > catastrophicThreshold),
						cloud.totalMass,
						cloud.debrisCount);
	// TODO add altitude logging

	currentPopulation.eventLog.push_back(tempEvent);

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