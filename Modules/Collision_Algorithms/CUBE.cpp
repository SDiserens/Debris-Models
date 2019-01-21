// CUBE.cpp : contains the implementation of the CUBE algorithm.
//

#include "stdafx.h"
#include "CUBE.h"

CUBEApproach::CUBEApproach(double dimension, bool probabilities)
{
	cubeDimension = dimension;
	cubeVolume = dimension * dimension * dimension;
	outputProbabilities = probabilities;
}

void FilterRecursion(vector<pair<long, long>>& pairList, vector<pair<long, long>> hashList, int i, int step);

void CUBEApproach::mainCollision(DebrisPopulation& population, double timeStep)
{
	double tempProbability, collisionRate;
	DebrisObject objectI, objectJ;
	vector<pair<long, long>> pairList;

	map<long, tuple<int, int, int>> cubeIDList;
	tuple<int, int, int> cube;
	double M;

	// For each object in population -
	for ( pair<long, DebrisObject> debris : population.population)
	{
		//	-- Generate Mean anomaly (randomTau)
		M = randomNumberTau();
		debris.second.SetMeanAnomaly(M);
		// TODO is this persistent outside of loop

		//	-- Calculate position & Identify CUBE ID
		cube = IdentifyCube(debris.second.GetPosition());

		//	-- Store Cube ID
		cubeIDList.emplace(debris.first, cube);
	}

	// Filter Cube List
	pairList = CubeFilter(cubeIDList);

	// For each conjunction (cohabiting pair)
	for (pair<long, long> &collisionPair : pairList)
	{
		//	-- Calculate collision rate in cube
		collisionRate = CollisionRate(population.GetObject(collisionPair.first),
									  population.GetObject(collisionPair.second));
		tempProbability = timeStep * collisionRate;

		//	-- Determine if collision occurs through MC (random number generation)
		if (outputProbabilities)
		{
			//	-- Store collision probability
			collisionProbabilities.push_back(tempProbability);
			collisionList.push_back(collisionPair);
		}
		else if (DetermineCollision(tempProbability))
			// Store Collisions 
			collisionList.push_back(collisionPair);
	}
}

void CUBEApproach::SwitchGravityComponent()
{
	relativeGravity = !relativeGravity;
}

long CUBEApproach::PositionHash(tuple<int, int, int> position)
{
	long hash;
	// XORT Vector Hash Function
	hash = ((get<0>(position) * p1) ^ (get<1>(position) * p2) ^ (get<2>(position) * p3)) % 10000;

	return hash;
}

vector<pair<long, long>> CUBEApproach::CubeFilter(map<long, tuple<int, int, int>> cubeIDList)
{
	vector<pair<long, long>> pairList;
	vector<pair<long, long>> hashList;
	long hash, ID1, ID2;
	//TODO - Cube Filter
	// Filter CUBEIDs
	for (auto cubeID : cubeIDList)
	{
		//	-- Hash ID
		hash = PositionHash(cubeID.second);
		hashList.push_back(pair<long, long>(hash, cubeID.first));
	}
	//	-- Sort Hash list : sorted by nature of being a map
	sort(hashList.begin(), hashList.end());

	for (int i=0; i+1 < hashList.size(); i++)
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
	for (int i = 0; i + 1 < pairList.size(); )
	{
		if (cubeIDList[pairList[i].first] != cubeIDList[pairList[i].second])
			pairList.erase(pairList.begin() + i);
		else
			++i;
	}
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

double CUBEApproach::CollisionRate(DebrisObject& objectI, DebrisObject& objectJ)
{
	double collisionCrossSection, boundingRadii, escapeVelocity2, gravitationalPerturbation;
	vector3D velocityI, velocityJ, relativeVelocity;

	velocityI = objectI.GetVelocity();
	velocityJ = objectJ.GetVelocity();

	relativeVelocity = velocityI.CalculateRelativeVector(velocityJ);
	boundingRadii = (objectI.GetRadius() + objectJ.GetRadius()) * 0.001; // Combined radii in kilometres

	if (relativeGravity)
	{
		escapeVelocity2 = 2 * (objectI.GetMass() + objectJ.GetMass()) * GravitationalConstant / boundingRadii;
		gravitationalPerturbation = (1 + escapeVelocity2 / relativeVelocity.vectorNorm2());
	}
	gravitationalPerturbation = 1;
	collisionCrossSection = gravitationalPerturbation * Pi * boundingRadii * boundingRadii;

	return  collisionCrossSection * relativeVelocity.vectorNorm() / cubeVolume;
}

tuple<int, int, int> CUBEApproach::IdentifyCube(vector3D& position)
{
	int X, Y, Z;

	X = int(position.x / cubeDimension);
	Y = int(position.y / cubeDimension);
	Z = int(position.z / cubeDimension);

	return tuple<int, int, int>(X, Y, Z);
}

bool CUBEApproach::DetermineCollision(double collisionProbability)
{
	return false;
}