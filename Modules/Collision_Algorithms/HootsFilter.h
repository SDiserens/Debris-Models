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
	bool PerigeeApogeeTest(CollisionPair objectPair);
	bool GeometricFilter(CollisionPair objectPair);
	vector<double> TimeFilter(CollisionPair objectPair, double timeStep);
	vector<double> CoplanarFilter(CollisionPair objectPair, double timeStep);
	vector<double> DetermineCollisionTimes(CollisionPair objectPair, vector<double> candidateTimeList);

	//Secondary functions
	vector<pair<double, double>> CalculateTimeWindows(pair<double,double> window, pair<double, double> window2, double period);
	double CalculateClosestApproachTime(CollisionPair objectPair, double candidateTime);
	double CalculateFirstDerivateSeparation(CollisionPair objectPair, double candidateTime);
	double CalculateSecondDerivativeSeparation(CollisionPair objectPair, double candidateTime);
};

