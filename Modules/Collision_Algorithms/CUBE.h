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
	bool offset = false; 

public:
	CUBEApproach(bool probabilities = false, double dimension = 10, int runs=1, bool switchOffset=false);

	void MainCollision(DebrisPopulation& population, double timeStep);
	void MainCollision_P(DebrisPopulation& population, double timeStep);
	void SetThreshold(double threshold);
	void SetMOID(int moid);

protected:
	double CollisionRate(CollisionPair& objectPair);
	list<CollisionPair> CreatePairList(DebrisPopulation& population);
	list<CollisionPair> CreatePairList_P(DebrisPopulation& population);
	list<CollisionPair> CreateOffsetPairList(DebrisPopulation& population);


	long PositionHash(tuple<int, int, int>);
	tuple<int, int, int> IdentifyCube(vector3D& position);
	tuple<int, int, int> IdentifyOffsetCube(vector3D& position);
	list<CollisionPair> CubeFilter(unordered_map<long, tuple<int, int, int>> cubeIDList);
	list<CollisionPair> OffsetCubeFilter(unordered_map<long, vector<tuple<int, int, int>>> cubeIDList);

};