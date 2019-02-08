// OrbitTrace.cpp : contains the implementation of the Orbit Trace collision algorithm.
//

#include "stdafx.h"
#include "OrbitTrace.h"


void OrbitTrace::MainCollision(DebrisPopulation& population, double timeStep)
{
}

double OrbitTrace::CollisionRate(DebrisObject & objectI, DebrisObject & objectJ)
{
	return 0.0;
}

vector<pair<long, long>> OrbitTrace::CreatePairList(DebrisPopulation & population)
{
	return vector<pair<long, long>>();
}
