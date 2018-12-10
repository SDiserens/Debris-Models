// Includes classes specific to the NASA Standard Breakup Model
//
#pragma once
#include "fragmentation.h"

class NSBMFragmentCloud : public FragmentCloud
{
public:
	bool explosion;
public:
	NSBMFragmentCloud(bool explosion, double minLength, double maxLength);
	int NumberOfFragments(double length);

private:
};

class NSBMDebrisFragment : public DebrisObject
{
public:
	double lambda, chi;
	bool explosion;
	// A/m Distribution parameters
	double alpha, mu_1, sigma_1, mu_2, sigma_2;

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