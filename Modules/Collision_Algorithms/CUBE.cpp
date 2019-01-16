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

void CUBEApproach::mainCollision(DebrisPopulation& population, double timeStep)
{
	double tempProbability, collisionRate;
	DebrisObject objectI, objectJ;
	vector<pair<long, long>> pairList;

	// For each object in population -
	//	-- Generate Mean anomaly (randomTau)
	//	-- Calculate position
	//	-- Identify CUBE ID

	pairList = CubeFilter();

	// For each conjunction (cohabiting pair)
	for (auto &collisionPair : pairList)
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

double CUBEApproach::PositionHash(vector3D position)
{
	// TODO - Vector Hash Function
	return 0.0;
}

vector<pair<long, long>> CUBEApproach::CubeFilter()
{
	//TODO - Cube Filter
	// Filter CUBEIDs
	//	-- Hash ID
	//	-- Sort Hash list
	//	-- Identify duplicate hash values
	//	-- (Sanitize results to remove hash clashes)

	return vector<pair<long, long>>();
}

double CUBEApproach::CollisionRate(DebrisObject objectI, DebrisObject objectJ)
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

bool CUBEApproach::DetermineCollision(double collisionProbability)
{
	return false;
}