
#include "stdafx.h"
#include "Collisions.h"


void CollisionAlgorithm::MainCollision(DebrisPopulation& population, double timeStep)
{
	double tempProbability, collisionRate;
	vector<pair<long, long>> pairList;
	// Filter Cube List
	pairList = CreatePairList(population);

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
			newCollisionProbabilities.push_back(tempProbability);
			newCollisionList.push_back(collisionPair);
		}
		else
		{
			if (DetermineCollision(tempProbability))
			{
				// Store Collisions 
				collisionList.push_back(collisionPair);
				newCollisionList.push_back(collisionPair);
			}
		}
	}
	elapsedTime += timeStep;
}

vector<pair<long, long>> CollisionAlgorithm::GetCollisionList()
{
	return collisionList;
}

vector<double> CollisionAlgorithm::GetCollisionProbabilities()
{
	return collisionProbabilities;
}

vector<pair<long, long>> CollisionAlgorithm::GetNewCollisionList()
{
	vector<pair<long, long>> newList(newCollisionList);
	newCollisionList.clear();
	return newList;
}

vector<double> CollisionAlgorithm::GetNewCollisionProbabilities()
{
	vector<double> newList(newCollisionProbabilities);
	newCollisionProbabilities.clear();
	return newList;
}

bool CollisionAlgorithm::DetermineCollision(double collisionProbability)
{
	return randomNumber() < collisionProbability;
}


void CollisionAlgorithm::SwitchGravityComponent()
{
	relativeGravity = !relativeGravity;
}
