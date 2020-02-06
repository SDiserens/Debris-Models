// CUBE.cpp : contains the implementation of the CUBE algorithm.
//

#include "stdafx.h"
#include "CUBE.h"

CUBEApproach::CUBEApproach(bool probabilities, double dimension, int runs, bool switchOffset)
{
	cubeDimension = dimension;
	cubeVolume = dimension * dimension * dimension;
	outputProbabilities = probabilities;
	mcRuns = runs;
	offset = switchOffset;
	elapsedTime = 0;
}

void FilterRecursion(vector<CollisionPair>& pairList, vector<pair<long, long>> hashList, int i, int step);
void FilterRecursionOffset(map<pair<long, long>, int>& pairCount,	vector<pair<long, pair<long, int>>> hashList, map<long, vector<tuple<int, int, int>>> cubeIDList, int i, int step);

void CUBEApproach::SetThreshold(double threshold)
{
	cubeDimension = threshold;
	cubeVolume = threshold * threshold * threshold;
}

void CUBEApproach::MainCollision(DebrisPopulation& population, double timeStep)
{
	double tempProbability, collisionRate, altitude, mass, adjustment;
	vector<CollisionPair> pairList;
	// Filter Cube List
	for (int j = 0; j < mcRuns; j++)
	{
		if (offset) {
			pairList = CreateOffsetPairList(population);
			// Should this probability be divided by 8?
			adjustment = timeStep / mcRuns;
		}
		else {
			pairList = CreatePairList(population);
			adjustment = timeStep / mcRuns;
		}

		// For each conjunction (cohabiting pair)
		for (CollisionPair &collisionPairID : pairList)
		{
			pair<long, long> pairID(collisionPairID.primaryID, collisionPairID.secondaryID);
			CollisionPair collisionPair(population.GetObject(pairID.first), population.GetObject(pairID.second));
			//	-- Calculate collision rate in cube
			collisionRate = CollisionRate(collisionPair);
			tempProbability = adjustment * collisionRate;

			altitude = collisionPair.primary.GetElements().GetRadialPosition();
			mass = collisionPair.primary.GetMass() + collisionPair.secondary.GetMass();
			Event tempEvent(population.GetEpoch(), pairID.first, pairID.second, collisionPair.GetRelativeVelocity(), mass, altitude);
			//	-- Determine if collision occurs through MC (random number generation)
			if (outputProbabilities)
			{
				//	-- Store collision probability
				//collisionProbabilities.push_back(tempProbability);
				//collisionList.push_back(collisionPair);
				newCollisionProbabilities.push_back(tempProbability);
				newCollisionList.push_back(tempEvent);
			}
			else
			{
				if (DetermineCollision(tempProbability))
				{
					// Store Collisions 
					collisionList.push_back(tempEvent);
					newCollisionList.push_back(tempEvent);
				}
			}
		}
	}
	elapsedTime += timeStep;
}

vector<CollisionPair> CUBEApproach::CreatePairList(DebrisPopulation& population)
{
	tuple<int, int, int> cube;
	double M;
	map<long, tuple<int, int, int>> cubeIDList;
	vector<CollisionPair> pairList;
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

vector<CollisionPair> CUBEApproach::CreateOffsetPairList(DebrisPopulation& population)
{
	tuple<int, int, int> cube;
	double M;
	map<long, vector<tuple<int, int, int>>> cubeIDList;
	vector<tuple<int, int, int>> cubes;
	vector<CollisionPair> pairList;
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
		cube = IdentifyOffsetCube(debris.second.GetPosition());

		//	-- Store Cube IDs
		// x+, y+, z+
		cubes.push_back(make_tuple(get<0>(cube) + 1, get<1>(cube) + 1, get<2>(cube) + 1));
		// x+, y+, z-
		cubes.push_back(make_tuple(get<0>(cube) + 1, get<1>(cube) + 1, get<2>(cube)));
		// x+, y-, z+
		cubes.push_back(make_tuple(get<0>(cube) + 1, get<1>(cube), get<2>(cube) + 1));
		// x+, y-, z-
		cubes.push_back(make_tuple(get<0>(cube) + 1, get<1>(cube), get<2>(cube)));
		// x-, y+, z+
		cubes.push_back(make_tuple(get<0>(cube), get<1>(cube) + 1, get<2>(cube) + 1));
		// x-, y+, z-
		cubes.push_back(make_tuple(get<0>(cube), get<1>(cube) + 1, get<2>(cube)));
		// x-, y-, z+
		cubes.push_back(make_tuple(get<0>(cube), get<1>(cube), get<2>(cube) + 1));
		// x-, y-, z-
		cubes.push_back(make_tuple(get<0>(cube), get<1>(cube), get<2>(cube)));

		cubeIDList.emplace(debris.first, cubes);

	}
	return OffsetCubeFilter(cubeIDList);
}

double CUBEApproach::CollisionRate(CollisionPair& objectPair)
{
	double collisionCrossSection, relativeVelocity;
	vector3D velocityI, velocityJ;

	velocityI = objectPair.primary.GetVelocity();
	velocityJ = objectPair.secondary.GetVelocity();

	relativeVelocity = velocityI.CalculateRelativeVector(velocityJ).vectorNorm();
	objectPair.SetRelativeVelocity(relativeVelocity);
	collisionCrossSection = CollisionCrossSection(objectPair.primary, objectPair.secondary);

	return  collisionCrossSection * relativeVelocity * objectPair.overlapCount / cubeVolume;
}

long CUBEApproach::PositionHash(tuple<int, int, int> position)
{
	long hash;
	// XORT Vector Hash Function
	hash = ((get<0>(position) * p1) ^ (get<1>(position) * p2) ^ (get<2>(position) * p3)) % 1000000000;

	return hash;
}

vector<CollisionPair> CUBEApproach::CubeFilter(map<long, tuple<int, int, int>> cubeIDList)
{
	vector<CollisionPair> pairList;
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
			pairList.push_back(CollisionPair(ID1, ID2));
			pairList.back().overlapCount = 1;
			if (i != 0 && (hashList[i].first == hashList[i - 1].first))
				FilterRecursion( pairList, hashList, i+1, 2);
		}
	
	}
	//	-- (Sanitize results to remove hash clashes)
	if (pairList.size() != 0)
 		pairList.erase( remove_if( pairList.begin(), pairList.end(),
						[&](CollisionPair cubePair) {return cubeIDList[cubePair.primaryID] != cubeIDList[cubePair.secondaryID]; } )
						, pairList.end());

 	return pairList;
}

vector<CollisionPair> CUBEApproach::OffsetCubeFilter(map<long, vector<tuple<int, int, int>>> cubeIDList)
{
	vector<CollisionPair> pairList;
	map<pair<long, long>, int> pairCount;
	vector<pair<long, pair<long, int>>> hashList;
	long hash, ID1, ID2;
	int i, position1, position2;
	// Cube Filter
	// Filter CUBEIDs
	for (auto cubeList : cubeIDList)
	{
		i = 0;
		for (auto cubeID : cubeList.second) {
			//	-- Hash ID
			hash = PositionHash(cubeID);
			hashList.push_back(make_pair(hash, make_pair(cubeList.first, i++)));
		}
	}
	//	-- Sort Hash list : sorted by nature of being a map
	sort(hashList.begin(), hashList.end());

	for (i = 0; i + 1 < hashList.size(); i++)
	{
		//	-- Identify duplicate hash values
		if (hashList[i].first == hashList[i + 1].first)
		{
			ID1 = hashList[i].second.first;
			position1 = hashList[i].second.second;
			ID2 = hashList[i + 1].second.first;
			position2 = hashList[i + 1].second.second;

			//	-- (Sanitize results to remove hash clashes)
			if (cubeIDList[ID1][position1] == cubeIDList[ID2][position2]) {
				++pairCount[make_pair(ID1, ID2)];
			}

			if (i != 0 && (hashList[i].first == hashList[i - 1].first))
				FilterRecursionOffset(pairCount, hashList, cubeIDList, i + 1, 2);
		}

	}

	for (auto pair : pairCount) {
		pairList.push_back(CollisionPair(pair.first.first, pair.first.second));
		pairList.back().overlapCount = pair.second;
	}
	return pairList;
}

void FilterRecursion(vector<CollisionPair>& pairList, vector<pair<long, long>> hashList, int i, int step)
{
	long ID1, ID2;
	ID1 = hashList[i - step].second;
	ID2 = hashList[i ].second;
	pairList.push_back(CollisionPair(ID1, ID2));
	pairList.back().overlapCount = 1;
	if (i - step != 0 && (hashList[i].first == hashList[i - step - 1].first))
		FilterRecursion(pairList, hashList, i,  step +1 );
}

void FilterRecursionOffset(map<pair<long, long>, int>& pairCount, vector<pair<long, pair<long, int>>> hashList, map<long, vector<tuple<int, int, int>>> cubeIDList, int i, int step)
{
	long ID1, ID2;
	int position1, position2;
	ID1 = hashList[i-step].second.first;
	position1 = hashList[i-step].second.second;
	ID2 = hashList[i].second.first;
	position2 = hashList[i].second.second;

	//	-- (Sanitize results to remove hash clashes)
	if (cubeIDList[ID1][position1] == cubeIDList[ID2][position2]) {
		++pairCount[make_pair(ID1, ID2)];
	}

	if (i - step != 0 && (hashList[i].first == hashList[i - step - 1].first))
		FilterRecursionOffset(pairCount, hashList, cubeIDList, i, step + 1);
}

tuple<int, int, int> CUBEApproach::IdentifyCube(vector3D& position)
{
	int X, Y, Z;

	X = int(position.x / cubeDimension);
	Y = int(position.y / cubeDimension);
	Z = int(position.z / cubeDimension);

	return tuple<int, int, int>(X, Y, Z);
}

tuple<int, int, int> CUBEApproach::IdentifyOffsetCube(vector3D& position)
{
	int X, Y, Z;

	X = int(position.x / cubeDimension - 0.5);
	Y = int(position.y / cubeDimension - 0.5);
	Z = int(position.z / cubeDimension - 0.5);

	return tuple<int, int, int>(X, Y, Z);
}
