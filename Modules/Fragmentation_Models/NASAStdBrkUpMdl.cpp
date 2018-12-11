// NASAStdBrkUpMdl.cpp : Contains the implementation of the original NASA Standard Breakup Model.
//

#include "stdafx.h"
#include "fragmentation.h"
#include "NSBM.h"

int mainBreakup(DebrisPopulation& population, DebrisObject& targetObject, DebrisObject *projectilePointer=NULL, float minLength=0.001)
{
    // Initialise Variables
    bool explosion;
	FragmentCloud targetDebrisCloud, projectileDebrisCloud;

	// Store relevant object variables
    
	if (projectilePointer == NULL)
	{
		explosion = true;
		targetDebrisCloud = NSBMFragmentCloud(targetObject, minLength);
		MergeFragmentPopulations(population, targetDebrisCloud);
	}

	else
	{
		explosion = false;
		DebrisObject& projectileObject = *projectilePointer;
		delete projectilePointer;
		targetDebrisCloud = NSBMFragmentCloud(targetObject, projectileObject, minLength);
		projectileDebrisCloud = NSBMFragmentCloud(projectileObject, targetObject, minLength);
		MergeFragmentPopulations(population, targetDebrisCloud);
		MergeFragmentPopulations(population, projectileDebrisCloud);
	}

	return 0;
}


NSBMFragmentCloud::NSBMFragmentCloud(DebrisObject& targetObject, double minLength) //Create an explosion Cloud
{
	
	double totalMass, maxLength, assignedMass;

	// Identify key variables
	totalMass = targetObject.GetMass();
	maxLength = targetObject.GetLength();

	SetNumberFragmentParametersExplosion();
	SetNumberOfFragments(CalculateNumberOfFragments(minLength));

	// Identify different size buckets for fragments

	// Create ExplosionCloud for each bucket

		// Generate debris in each bucket

		// Update overall FragmentCloud variables

	// Check conservation of Mass, Momentum and Energy within limits

}

NSBMFragmentCloud::NSBMFragmentCloud(DebrisObject& targetObject, DebrisObject& projectileObject, double minLength) //Create a collision Cloud
{
	
	double totalMass, maxLength, assignedMass, kineticEnergy, energyMassRatio, impactMass;

	// Identify key variables
	totalMass = targetObject.GetMass();
	maxLength = targetObject.GetLength();
	impactMass = projectileObject.GetMass();
	vector3D velocity = targetObject.GetVelocity();
	vector3D relativeVelocity = velocity.CalculateRelativeVector(projectileObject.GetVelocity());
	kineticEnergy = CalculateKineticEnergy(relativeVelocity, projectileObject.GetMass());
	energyMassRatio = CalculateEnergyToMass(kineticEnergy, totalMass);

	if (energyMassRatio > 40)
		SetNumberFragmentParametersCatastrophicCollision();
	else
		SetNumberFragmentParametersCollision();

	SetNumberOfFragments(CalculateNumberOfFragments(minLength));

	// Identify different size buckets for fragments

	// Create CollisionCloud for each bucket
		
		// Generate debris in each bucket

		// Update overall FragmentCloud variables

	// Check conservation of Mass, Momentum and Energy within limits

}


//DebrisObject GenerateDebrisObject()
NSBMFragmentCloud::NSBMFragmentCloud(bool init_explosion, double init_minLength, double init_maxLength) // Generates a fragment size bucket
{
	explosion = init_explosion;
	maxLength = init_maxLength;
	minLength = init_minLength;
}

void NSBMFragmentCloud::GenerateDebrisFragments()
{
	// Assign fragment length

	// Calculate area-to-mass ratio

	// Calculate mass

	// Calculate Velocity

	// Identify updated orbital elements

	// Create new DebrisObject

	// Update FragmentCloud variables for bucket
}

int NSBMFragmentCloud::CalculateNumberOfFragments(double length)
{
	int numFrag = nFragCoefficient * pow(length, nFragExponent);
	return numFrag;
}

int NSBMFragmentCloud::CalculateBucketFragments(double lowerLength, double upperLength)
{
	int nFrag = CalculateNumberOfFragments(lowerLength) - CalculateNumberOfFragments(upperLength);
	return nFrag;
}

void NSBMFragmentCloud::SetNumberFragmentParametersExplosion()
{
	nFragExponent = -1.6;
	nFragCoefficient = scaling * 6;
}

void NSBMFragmentCloud::SetNumberFragmentParametersCollision()
{
	double ejectaMass = impactMass * 0.000001 * relativeVelocity.vectorNorm2();
	nFragCoefficient = 0.1 * pow(ejectaMass, 0.75);
	nFragExponent = -1.71;
}

void NSBMFragmentCloud::SetNumberFragmentParametersCatastrophicCollision()
{

	double ejectaMass = totalMass;
	nFragCoefficient = 0.1 * pow(ejectaMass, 0.75);
	nFragExponent = -1.71;
}

void NSBMFragmentCloud::SetNumberOfFragments(int nFrag)
{
	int numFrag = nFrag;
}


// Debris Fragments
NSBMDebrisFragment::NSBMDebrisFragment(double init_length, bool init_explosion)
{
	length = init_length;
	lambda = log10(length);
	explosion = init_explosion;
	CalculateArea();
	
	//SetSmallAreaMassParameters()

	if (explosion)
	{
		sourceType = 1;
		SetExplosionAreaMassParameters();
	}
	else
	{
		sourceType = 2;
		SetCollisionAreaMassParameters();
	}
	GenerateAreaToMassValue();
	CalculateMassFromArea();
}

NSBMDebrisFragment::NSBMDebrisFragment(double init_length, double init_mass, bool init_explosion)
{
	length = init_length;
	lambda = log10(length);
	mass = init_mass;
	explosion = init_explosion;
	CalculateArea();

	if (explosion)
	{
		sourceType = 1;
		//SetExplosionAreaMassParameters();
	}
	else
	{
		sourceType = 2;
		//SetCollisionAreaMassParameters();
	}
	CalculateAreaToMass();
}

void NSBMDebrisFragment::SetExplosionAreaMassParameters()
{
	// alpha
	if (lambda <= -1.4)
		alpha = 1;
	else if (lambda >= 0)
		alpha = 0.5;
	else
		alpha = 1 - 0.3571 * (lambda + 1.4);

	// mu_1
	if (lambda <= -0.5)
		mu_1 = -0.45;
	else if (lambda >= 0)
		mu_1 = -0.9;
	else
		mu_1 = -0.45 - 0.9 * (lambda + 0.5);

	// sigma_1
	sigma_1 = 0.55;

	// mu_2
	mu_2 = -0.9;

	// sigma_2
	if (lambda <= -1.0)
		sigma_2 = 0.28;
	else if (lambda >= 0.1)
		sigma_2 = 0.1;
	else
		sigma_2 = 0.28 - 0.1636 * (lambda + 1);
}

void NSBMDebrisFragment::SetCollisionAreaMassParameters()
{
	// alpha
	if (lambda <= -1.95)
		alpha = 0;
	else if (lambda >= 0.55)
		alpha = 1.0;
	else
		alpha = 0.3 + 0.4 * (lambda + 1.2);

		// mu_1
	if (lambda <= -1.1)
		mu_1 = -0.6;
	else if (lambda >= 0)
		mu_1 = -0.95;
	else
		mu_1 = -0.6 - 0.318 * (lambda + 1.1);

		// sigma_1
	if (lambda <= -1.3)
		sigma_1 = 0.1;
	else if (lambda >= -0.3)
		sigma_1 = 0.3;
	else
		sigma_1 = 0.1 + 0.2 * (lambda + 1.3);

		// mu_2
	if (lambda <= 0.7)
		mu_2 = -1.2;
	else if (lambda >= -0.1)
		mu_2 = -2.0;
	else
		mu_2 = -1.2 - 1.333 * (lambda + 0.7);

		// sigma_2
	if (lambda <= -0.5)
		sigma_2 = 0.5;
	else if (lambda >= -0.3)
		sigma_2 = 0.3;
	else
		sigma_2 = 0.5 - 1 * (lambda + 0.5);
}

void NSBMDebrisFragment::SetSmallAreaMassParameters()
{
	alpha = 1;

	if (lambda <= -1.75)
		mu_1 = -0.3;
	else if (lambda >= -1.25)
		mu_1 = -1.0;
	else
		mu_1 = -0.3 - 1.4 * (lambda + 1.75);

	if (lambda <= -3.5)
		sigma_1 = 0.2;
	else
		sigma_1 = 0.2 + 0.1333 * (lambda + 3.5);

	mu_2 = 0;
	sigma_2 = 1;
}

void NSBMDebrisFragment::GenerateAreaToMassValue()
{

}


void NSBMDebrisFragment::CalculateArea()
{
	if (length < 0.00167)
		area = 0.540424 * length * length;
	else
		area = 0.556945 * pow(length, 2.0047077);
}

void NSBMDebrisFragment::CalculateVelocity()
{

}

