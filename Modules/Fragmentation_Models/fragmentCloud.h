#pragma once
class FragmentCloud
{
public:
	// Initialise object variables
	int debrisCount;
	double totalMass, minLength, maxLength;
	
	// Object Constructors
	FragmentCloud(double minLength, double maxLength);
	~FragmentCloud();

	// Pre-define object functions
	void AddFragment(DebrisObject fragment);
	void AddCloud(FragmentCloud fragmentCloud);
};

