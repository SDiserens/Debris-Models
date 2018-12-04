// NASAStdBrkUpMdl.cpp : Contains the implementation of the original NASA Standard Breakup Model.
//

#include "stdafx.h"
#include "fragmentation.h"
#include "NSBM.h"

FragmentCloud GenerateExplosionDebris(DebrisObject& targetObject, float minLength);
FragmentCloud GenerateCollisionDebris(DebrisObject& targetObject, DebrisObject& projectileObject, float minLength);

int mainBreakup(DebrisPopulation& population, DebrisObject& targetObject, DebrisObject *projectilePointer=NULL, float minLength=0.001)
{
    // Initialise Variables
    bool explosion;
	FragmentCloud targetDebrisCloud, projectileDebrisCloud;

	// Store relevant object variables
    
	if (projectilePointer == NULL)
	{
		explosion = true;
		targetDebrisCloud = GenerateExplosionDebris(targetObject, minLength);
		MergeFragmentPopulations(population, targetDebrisCloud);
	}

	else
	{
		explosion = false;
		DebrisObject& projectileObject = *projectilePointer;
		delete projectilePointer;
		targetDebrisCloud = GenerateCollisionDebris(targetObject, projectileObject, minLength);
		projectileDebrisCloud = GenerateCollisionDebris(projectileObject, targetObject, minLength);
		MergeFragmentPopulations(population, targetDebrisCloud);
		MergeFragmentPopulations(population, projectileDebrisCloud);
	}


	return 0;
}

FragmentCloud GenerateExplosionDebris(DebrisObject& targetObject, float minLength)
{
	FragmentCloud debrisCloud;
	float totalMass, maxLength, assignedMass;

	// Identify key variables
	totalMass = targetObject.mass;
	maxLength = targetObject.length;

	// Create Cloud object

	// Identify different size buckets for fragments

	// Create ExplosionCloud for each bucket

		// Generate debris in each bucket
		
			// Assign fragment length

			// Calculate area-to-mass ratio

			// Calculate mass

			// Calculate Velocity

			// Identify updated orbital elements

			// Create new DebrisObject

		// Update FragmentCloud variables for bucket

	// Update overall FragmentCloud variables

	// Check conservation of Mass, Momentum and Energy within limits

	return debrisCloud;
}

FragmentCloud GenerateCollisionDebris(DebrisObject& targetObject, DebrisObject& projectileObject, float minLength)
{
	FragmentCloud debrisCloud;
	float totalMass, maxLength, assignedMass, kineticEnergy, energyMassRatio;

	// Identify key variables
	totalMass = targetObject.mass;
	maxLength = targetObject.length;
	kineticEnergy = 0;
	energyMassRatio = 0;

	// Create Cloud object

	// Identify different size buckets for fragments

	// Create CollisionCloud for each bucket

		// Generate debris in each bucket

			// Assign fragment length

			// Calculate area-to-mass ratio

			// Calculate mass

			// Calculate Velocity

			// Identify updated orbital elements

			// Create new DebrisObject

		// Update FragmentCloud variables for bucket

	// Update overall FragmentCloud variables

	// Check conservation of Mass, Momentum and Energy within limits

	return debrisCloud;
}


//DebrisObject GenerateDebrisObject()


NSBMFragmentCloud::NSBMFragmentCloud(bool explosion, double minLength, double maxLength)
{

	
}

void NSBMFragmentCloud::SetExplosionAreaMassParameters()
{
}

void NSBMFragmentCloud::SetCollisionAreaMassParameters()
{
}

int NSBMFragmentCloud::NumberOfFragments(double length)
{

}

NSBMDebrisFragment::NSBMDebrisFragment(double init_length)
{

}

NSBMDebrisFragment::NSBMDebrisFragment(double init_length, double init_mass)
{

}

void NSBMDebrisFragment::CalculateAreaToMass()
{

}

void NSBMDebrisFragment::CalculateArea()
{

}

void NSBMDebrisFragment::CalculateVelocity()
{

}

