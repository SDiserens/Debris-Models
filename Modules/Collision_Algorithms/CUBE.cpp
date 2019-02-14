// CUBE.cpp : contains the implementation of the CUBE algorithm.
//

#include "stdafx.h"
#include "CUBE.h"

CUBEApproach::CUBEApproach(double dimension, bool probabilities)
{
	cubeDimension = dimension;
	cubeVolume = dimension * dimension * dimension;
	outputProbabilities = probabilities;
	elapsedTime = 0;
}

void FilterRecursion(vector<pair<long, long>>& pairList, vector<pair<long, long>> hashList, int i, int step);

vector<pair<long, long>> CUBEApproach::CreatePairList(DebrisPopulation& population)
{
	tuple<int, int, int> cube;
	double M;
	map<long, tuple<int, int, int>> cubeIDList;
	// Prepare List
	cubeIDList.clear();

	// For each object in population -
	for (auto& debris : population.population)
	{
		//	-- Generate Mean anomaly (randomTau)
		M = randomNumberTau();
		debris.second.SetMeanAnomaly(M);
		// is this persistent outside of loop? - Does it need to be - Yes for velocity calculation - fixed with reference auto&

		//	-- Calculate position & Identify CUBE ID
		cube = IdentifyCube(debris.second.GetPosition());

		//	-- Store Cube ID
		cubeIDList.emplace(debris.first, cube);
	}

	return CubeFilter(cubeIDList);
}


double CUBEApproach::CollisionRate(DebrisObject& objectI, DebrisObject& objectJ)
{
	double collisionCrossSection;
	vector3D velocityI, velocityJ, relativeVelocity;

	velocityI = objectI.GetVelocity();
	velocityJ = objectJ.GetVelocity();

	relativeVelocity = velocityI.CalculateRelativeVector(velocityJ);

	collisionCrossSection = CollisionCrossSection(objectI, objectJ);

	return  collisionCrossSection * relativeVelocity.vectorNorm() / cubeVolume;
}

long CUBEApproach::PositionHash(tuple<int, int, int> position)
{
	long hash;
	// XORT Vector Hash Function
	hash = ((get<0>(position) * p1) ^ (get<1>(position) * p2) ^ (get<2>(position) * p3)) % 1000000000;

	return hash;
}

vector<pair<long, long>> CUBEApproach::CubeFilter(map<long, tuple<int, int, int>> cubeIDList)
{
	vector<pair<long, long>> pairList;
	vector<pair<long, long>> hashList;
	long hash, ID1, ID2;
	int i;
	// Cube Filter
	// Filter CUBEIDs
	for (auto cubeID : cubeIDList)
	{
		//	-- Hash ID
		hash = PositionHash(cubeID.second);
		hashList.push_back(pair<long, long>(hash, cubeID.first));
	}
	//	-- Sort Hash list : sorted by nature of being a map
	sort(hashList.begin(), hashList.end());

	for (i=0; i+1 < hashList.size(); i++)
	{
		//	-- Identify duplicate hash values
		if (hashList[i].first == hashList[i + 1].first)
		{
			ID1 = hashList[i].second;
			ID2 = hashList[i + 1].second;
			pairList.push_back(pair<long, long>(ID1, ID2));

			if (i != 0 && (hashList[i].first == hashList[i - 1].first))
				FilterRecursion( pairList, hashList, i+1, 2);
		}
		
	}
	//	-- (Sanitize results to remove hash clashes)
	if (pairList.size() != 0)
 		pairList.erase( remove_if( pairList.begin(), pairList.end(),
						[&](pair<long, long> cubePair) {return cubeIDList[cubePair.first] != cubeIDList[cubePair.second]; } )
						, pairList.end());

 	return pairList;
}


void FilterRecursion(vector<pair<long, long>>& pairList, vector<pair<long, long>> hashList, int i, int step)
{
	long ID1, ID2;
	ID1 = hashList[i - step].second;
	ID2 = hashList[i ].second;
	pairList.push_back(pair<long, long>(ID1, ID2));
	if (i - step != 0 && (hashList[i].first == hashList[i - step - 1].first))
		FilterRecursion(pairList, hashList, i,  step +1 );
}


tuple<int, int, int> CUBEApproach::IdentifyCube(vector3D& position)
{
	int X, Y, Z;

	X = int(position.x / cubeDimension);
	Y = int(position.y / cubeDimension);
	Z = int(position.z / cubeDimension);

	return tuple<int, int, int>(X, Y, Z);
}
