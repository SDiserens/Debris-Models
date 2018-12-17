#pragma once
#include <vector>

class FragmentCloud
{
public:
	// Initialise object variables
	int debrisCount, nBuckets;
	double totalMass, averageMass, minLength, maxLength, averageLength, assignedMass, averageSpeed, totalKineticEnergy, averageKineticEnergy, totalVolume, averageVolume, averageDensity;
	std::vector<FragmentCloud> fragmentBuckets;
	std::vector<DebrisObject> fragments;
	vector3D relativeVelocity;
	vector3D velocity;
	vector3D averageVelocity;
	vector3D averageMomentum;
	//vector3D velocity;

	
	// Object Constructors
	FragmentCloud();
	FragmentCloud(double minLength, double maxLength, int buckets);
	~FragmentCloud();

	// Pre-define object functions
	void AddFragment(DebrisObject fragment);
	void AddCloud(FragmentCloud fragmentCloud);
};

