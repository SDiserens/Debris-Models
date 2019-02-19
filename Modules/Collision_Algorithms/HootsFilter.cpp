#include "stdafx.h"
#include "HootsFilter.h"


HootsFilter::HootsFilter(double init_conjThreshold, double init_collThreshold)
{
	conjunctionThreshold = init_conjThreshold;
	collisionThreshold = init_collThreshold;
}

void HootsFilter::MainCollision(DebrisPopulation & population, double timeStep)
{
}

double HootsFilter::CollisionRate(DebrisObject & objectI, DebrisObject & objectJ)
{
	return 0.0;
}

vector<pair<long, long>> HootsFilter::CreatePairList(DebrisPopulation & population)
{
	return vector<pair<long, long>>();
}

bool HootsFilter::PerigeeApogeeTest(CollisionPair objectPair)
{
	// TODO - Perigee Apogee Test
	return false;
}

bool HootsFilter::GeometricFilter(CollisionPair objectPair, double relativeInclination)
{
	// TODO - Geometric Filter
	return false;
}

bool HootsFilter::TimeFilter(CollisionPair objectPair, double relativeInclination, double timeStep)
{
	// TODO - Time Filter
	return false;
}

bool HootsFilter::CoplanarFilter(CollisionPair objectPair, double timeStep)
{
	// TODO - Coplanar Filter
	return false;
}

vector<double> HootsFilter::DetermineCollisionTimes(CollisionPair objectPair, vector<double> candidateTimeList)
{
	// TODO - Collision Times
	return vector<double>();
}

vector<pair<double, double>> HootsFilter::CalculateTimeWindows(pair<double, double> window, double period, double timestep)
{
	//TODO - Time windows
	return vector<pair<double, double>>();
}

double HootsFilter::CalculateClosestApproachTime(CollisionPair objectPair, double candidateTime)
{
	//TODO - closest approach time
	return 0.0;
}

double HootsFilter::CalculateFirstDerivateSeparation(CollisionPair objectPair, double candidateTime)
{
	//TODO - 1st derivative seperation
	return 0.0;
}

double HootsFilter::CalculateSecondDerivativeSeparation(CollisionPair objectPair, double candidateTime)
{
	// TODO - 2nd derivative seperation
	return 0.0;
}


