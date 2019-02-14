#pragma once

#include "stdafx.h"
#include "Collisions.h"

class CUBEApproach : public CollisionAlgorithm
{
protected:
	double cubeDimension, cubeVolume;
	int p1 = 73856093;
	int p2 = 19349663;
	int p3 = 83492791;
	//map<long, tuple<int, int, int>> cubeIDList;

public:
	CUBEApproach(double dimension, bool probabilities = false);


protected:
	double CollisionRate(DebrisObject& objectI, DebrisObject& objectJ);
	vector<pair<long, long>> CUBEApproach::CreatePairList(DebrisPopulation& population);
	long PositionHash(tuple<int, int, int>);
	tuple<int, int, int> IdentifyCube(vector3D& position);
	vector<pair<long,long>> CubeFilter(map<long, tuple<int, int, int>> cubeIDList);

};