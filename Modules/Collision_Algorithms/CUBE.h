#pragma once

#include "stdafx.h"
#include "Collisions.h"

class CUBEApproach
{
protected:
	double cubeDimension, cubeVolume;
	bool outputProbabilities;

public:
	CUBEApproach(double dimension, bool probabilities = false);
	void mainCollision(DebrisPopulation& population, double timeStep);

protected:
	double PositionHash(vector3D position);
	double CollisionRate(DebrisObject objectI, DebrisObject objectJ);
	vector<pair<long,long>> CubeFilter();
};