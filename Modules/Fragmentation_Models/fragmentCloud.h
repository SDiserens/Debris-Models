#pragma once
#include <vector>

class FragmentCloud
{
public:
	// Initialise object variables
	bool consMomentumFlag;
	int debrisCount, nBuckets;
	double totalMass, averageMass, minLength, maxLength, averageLength; // Working variables
	double assignedMass, averageSpeed, totalKineticEnergy, averageKineticEnergy, totalVolume, averageVolume, averageDensity, averageMomentumNorm; // Recorded variables
	
	// Working variables
	vector3D relativeVelocity;
	vector3D velocity;

	// Recorded vector variables
	vector3D averageVelocity, totalMomentum, averageMomentum;

	// Storage
	std::vector<FragmentCloud> fragmentBuckets;
	std::vector<DebrisObject> fragments;
	
	// Object Constructors
	FragmentCloud();
	FragmentCloud(double minLength, double maxLength, int buckets);
	~FragmentCloud();

	// Pre-define object functions
	void AddFragment(DebrisObject fragment);
	void AddCloud(FragmentCloud fragmentCloud);
};

