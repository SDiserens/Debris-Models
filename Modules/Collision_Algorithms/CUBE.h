#pragma once

#include "stdafx.h"
#include "Collisions.h"

class CUBEApproach : public CollisionAlgorithm
{
protected:
	double cubeDimension, cubeVolume, elapsedTime;
	bool outputProbabilities, relativeGravity = false;
	int p1 = 73856093;
	int p2 = 19349663;
	int p3 = 83492791;
	//map<long, tuple<int, int, int>> cubeIDList;

public:
	CUBEApproach(double dimension, bool probabilities = false);
	void SwitchGravityComponent();

	void MainCollision(DebrisPopulation& population, double timeStep);
	double GetElapsedTime();


protected:
	bool DetermineCollision(double collisionProbability);
	long PositionHash(tuple<int, int, int>);
	double CollisionRate(DebrisObject& objectI, DebrisObject& objectJ);
	tuple<int, int, int> IdentifyCube(vector3D& position);
	vector<pair<long,long>> CubeFilter(map<long, tuple<int, int, int>> cubeIDList);

};