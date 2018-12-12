// Includes classes specific to the NASA Standard Breakup Model
//
#pragma once
#include "fragmentation.h"

class NSBMFragmentCloud : public FragmentCloud
{
public:
	bool explosion;
	double scaling, impactMass, nFragExponent, nFragCoefficient;
	int nFrag;
	std::default_random_engine generator;

public:
	// Constructors
	NSBMFragmentCloud(DebrisObject& targetObject, double minLength); // Explosion contructor
	NSBMFragmentCloud(DebrisObject& targetObject, DebrisObject& projectileObject, double minLength); // Collision Constructor
	NSBMFragmentCloud(bool init_explosion, double init_minLength, double init_maxLength); // Bucket Constructor

	// Functions
	void SetNumberFragmentParametersExplosion();
	void SetNumberFragmentParametersCollision();
	void SetNumberFragmentParametersCatastrophicCollision();
	int CalculateNumberOfFragments(double length);
	int CalculateBucketFragments(double lowerLength, double upperLength);
	void SetNumberOfFragments(int nFrag);
	void GenerateDebrisFragments();
	double GenerateFragmentLength();

};

class NSBMDebrisFragment : public DebrisObject
{
public:
	double lambda, chi;
	bool explosion;
	// A/m Distribution parameters
	double alpha, mu_1, sigma_1, mu_2, sigma_2;
	std::default_random_engine generator;

public:
	NSBMDebrisFragment(double init_length, bool init_explosion);
	NSBMDebrisFragment(double init_length, double init_mass, bool init_explosion);
	void CalculateArea();
	void CalculateVelocity();
	void GenerateAreaToMassValue();

private:
	void SetSmallAreaMassParameters();
	void SetExplosionAreaMassParameters();
	void SetCollisionAreaMassParameters();
};