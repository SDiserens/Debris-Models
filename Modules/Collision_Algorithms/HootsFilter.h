#pragma once
#include "stdafx.h"
#include "Collisions.h"

class HootsFilter :
	public CollisionAlgorithm
{
	double conjunctionThreshold, collisionThreshold;

public:
	HootsFilter(double init_conjThreshold=10, double init_collThreshold=0.1);
	void MainCollision(DebrisPopulation& population, double timeStep);

protected:
	double CollisionRate(DebrisObject& objectI, DebrisObject& objectJ);
	vector<pair<long, long>> CreatePairList(DebrisPopulation& population);
	bool PerigeeApogeeTest(pair<DebrisObject&, DebrisObject&> objectPair);
	bool GeometricFilter(pair<DebrisObject&, DebrisObject&> objectPair, double relativeInclination);
	bool TimeFilter(pair<DebrisObject&, DebrisObject&> objectPair, double relativeInclination, double timeStep);
	bool CoplanarFilter(pair<DebrisObject&, DebrisObject&> objectPair, double timeStep);
	vector<double> DetermineCollisionTimes(pair<DebrisObject&, DebrisObject&> objectPair, vector<double> candidateTimeList);
};

