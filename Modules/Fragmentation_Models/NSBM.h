// Includes classes specific to the NASA Standard Breakup Model
//
#pragma once
#include "fragmentation.h"

class NSBMFragmentCloud : public FragmentCloud
{
public:
	NSBMFragmentCloud(bool explosion, double minLength, double maxLength);
	int NumberOfFragments(double length);

private:
	void SetExplosionAreaMassParameters();
	void SetCollisionAreaMassParameters();
};

class NSBMDebrisFragment : public DebrisObject
{
	NSBMDebrisFragment(double init_length);
	NSBMDebrisFragment(double init_length, double init_mass);
	void CalculateArea();
	void CalculateVelocity();
	void CalculateAreaToMass();
	void CalculateMass();
};