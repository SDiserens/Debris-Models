#pragma once

#include "stdafx.h"
#include "Collisions.h"



class CUBEApproach : public CollisionAlgorithm
{
protected:
	double cubeDimension, cubeVolume;
	int mcRuns;
	//map<long, tuple<int, int, int>> cubeIDList;
	int p1 = 73856093;
	int p2 = 19349663;
	int p3 = 83492791;

public:
	CUBEApproach(bool probabilities = false, double dimension = 10, int runs=1);

	void MainCollision(DebrisPopulation& population, double timeStep);

protected:
	double CollisionRate(CollisionPair& objectPair);
	vector<CollisionPair> CreatePairList(DebrisPopulation& population);


	long PositionHash(tuple<int, int, int>);
	tuple<int, int, int> IdentifyCube(vector3D& position);
	vector<CollisionPair> CubeFilter(map<long, tuple<int, int, int>> cubeIDList);

};