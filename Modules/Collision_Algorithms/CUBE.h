#pragma once

#include "stdafx.h"
#include "Collisions.h"

class CUBEApproach : protected CollisionAlgorithm
{
protected:
	double cubeDimension, cubeVolume;
	bool outputProbabilities, relativeGravity = false;

public:
	CUBEApproach(double dimension, bool probabilities = false);
	void SwitchGravityComponent();

	void mainCollision(DebrisPopulation& population, double timeStep);


protected:
	bool DetermineCollision(double collisionProbability);
	double PositionHash(vector3D position);
	double CollisionRate(DebrisObject objectI, DebrisObject objectJ);
	vector<pair<long,long>> CubeFilter();
};