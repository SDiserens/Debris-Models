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
	//Primary functions
	double CollisionRate(DebrisObject& objectI, DebrisObject& objectJ);
	vector<CollisionPair> CreatePairList(DebrisPopulation& population);
	bool PerigeeApogeeTest(CollisionPair objectPair);
	bool GeometricFilter(CollisionPair objectPair, double relativeInclination);
	bool TimeFilter(CollisionPair objectPair, double relativeInclination, double timeStep);
	bool CoplanarFilter(CollisionPair objectPair, double timeStep);
	vector<double> DetermineCollisionTimes(CollisionPair objectPair, vector<double> candidateTimeList);

	//Secondary functions
	vector<pair<double, double>> CalculateTimeWindows(pair<double,double> window, double period, double timestep);
	double CalculateClosestApproachTime(CollisionPair objectPair, double candidateTime);
	double CalculateFirstDerivateSeparation(CollisionPair objectPair, double candidateTime);
	double CalculateSecondDerivativeSeparation(CollisionPair objectPair, double candidateTime);
};

