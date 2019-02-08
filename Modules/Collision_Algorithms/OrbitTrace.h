#pragma once

#include "stdafx.h"
#include "Collisions.h"

class OrbitTrace : public CollisionAlgorithm
{
public:
	void MainCollision(DebrisPopulation& population, double timeStep);

protected:
	double CollisionRate(DebrisObject& objectI, DebrisObject& objectJ);
	vector<pair<long, long>> CreatePairList(DebrisPopulation& population);
};