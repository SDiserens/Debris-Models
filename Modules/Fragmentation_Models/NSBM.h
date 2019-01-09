// Includes classes specific to the NASA Standard Breakup Model
//
#pragma once
#include "fragmentation.h"

// Setting defaults for globals
extern int numFragBuckets;
extern string bridgingFunction;

int mainBreakup(DebrisPopulation& population, DebrisObject& targetObject, DebrisObject *projectilePointer = NULL, double minLength = 0.001);

class NSBMDebrisFragment : public DebrisObject
{
public:
	double lambda, chi, deltaVNorm, kineticEnergy, volume, density;
	bool explosion;
	// A/m Distribution parameters
	double alpha, mu_1, sigma_1, mu_2, sigma_2;
	vector3D deltaV;
	vector3D momentum;


public:
	NSBMDebrisFragment(double init_length, bool init_explosion, int source, int numFrag = 1);
	NSBMDebrisFragment(double init_length, double init_mass, bool init_explosion, int source, int numFrag = 1);
	void CalculateArea();
	void CalculateRelativeVelocity();
	void GenerateAreaToMassValue();
	void CalculateVolume();

	
private:
	void SetSmallAreaMassParameters();
	void AreaMassBridgingFunction();
	void SetUpperStageAreaMassParameters();
	void SetSpacecraftAreaMassParameters();
};

class NSBMFragmentCloud : public FragmentCloud
{
public:
	bool maxBucket;
	double scaling, impactMass, nFragExponent, nFragCoefficient;
	int numFrag;
	//std::default_random_engine generator;

public:
	// -- Constructors
	NSBMFragmentCloud(); // Default contructor
	NSBMFragmentCloud(DebrisObject& targetObject, double init_minLength); // Explosion contructor
	NSBMFragmentCloud(DebrisObject& targetObject, DebrisObject& projectileObject, double init_minLength); // Collision Constructor
	NSBMFragmentCloud(bool init_explosion, double init_minLength, double init_maxLength, int numFrag, double init_mass); // Bucket Constructor

	// -- Functions for breakup parameters
	void SetNumberFragmentParametersExplosion();
	void SetNumberFragmentParametersCollision();
	void SetNumberFragmentParametersCatastrophicCollision();

	// -- Functions For number of fragments
	int CalculateNumberOfFragments(double length);
	int CalculateBucketFragments(double lowerLength, double upperLength);
	void SetNumberOfFragments(int nFrag);

	// -- Functions for generating buckets and fragments
	void GenerateFragmentBuckets(DebrisObject& targetObject);
	void CreateFragmentBucket(DebrisObject& targetObject, double lowerLength, double upperLength);
	void CreateTopFragmentBucket(DebrisObject& targetObject, double lowerLength, double upperLength);
	void GenerateDebrisFragments(DebrisObject& targetObject);
	
	// -- Functions for updating cloud parameters
	void StoreFragmentVariables(NSBMDebrisFragment& tempFragment);
	void StoreFragmentVariables(NSBMFragmentCloud& tempFragmentCloud);
	void UpdateAverageVariables();
	
	// -- Functions for validating physical behaviour
	void ApplyConservationOfMass();
	void ApplyConservationOfMomentum();
	void ApplyConservationOfEnergy();
};
