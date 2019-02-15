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

bool HootsFilter::PerigeeApogeeTest(pair<DebrisObject&, DebrisObject&> objectPair)
{
	// TODO - Perigee Apogee Test
	return false;
}

bool HootsFilter::GeometricFilter(pair<DebrisObject&, DebrisObject&> objectPair, double relativeInclination)
{
	// TODO - Geometric Filter
	return false;
}

bool HootsFilter::TimeFilter(pair<DebrisObject&, DebrisObject&> objectPair, double relativeInclination, double timeStep)
{
	// TODO - Time Filter
	return false;
}

bool HootsFilter::CoplanarFilter(pair<DebrisObject&, DebrisObject&> objectPair, double timeStep)
{
	// TODO - Coplanar Filter
	return false;
}

vector<double> HootsFilter::DetermineCollisionTimes(pair<DebrisObject&, DebrisObject&> objectPair, vector<double> candidateTimeList)
{
	// TODO - Collision Times
	return vector<double>();
}

