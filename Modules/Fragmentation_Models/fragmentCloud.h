#pragma once
#include <vector>

class FragmentCloud
{
public:
	// Initialise object variables
	int debrisCount;
	double totalMass, minLength, maxLength;
	std::vector<FragmentCloud> fragmentBuckets;
	std::vector<DebrisObject> fragments;
	vector3D relativeVelocity;
	vector3D velocity;
	
	// Object Constructors
	FragmentCloud();
	FragmentCloud(double minLength, double maxLength, int buckets);
	~FragmentCloud();

	// Pre-define object functions
	void AddFragment(DebrisObject fragment);
	void AddCloud(FragmentCloud fragmentCloud);
};

