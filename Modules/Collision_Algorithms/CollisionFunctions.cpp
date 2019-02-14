
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

double CollisionAlgorithm::GetElapsedTime()
{
	return elapsedTime;
}
double CollisionAlgorithm::CollisionCrossSection(DebrisObject& objectI, DebrisObject& objectJ)
{
	double boundingRadii, escapeVelocity2, gravitationalPerturbation;
	vector3D velocityI = objectI.GetVelocity();
	vector3D velocityJ = objectJ.GetVelocity();

	vector3D relativeVelocity = velocityI.CalculateRelativeVector(velocityJ);
	boundingRadii = (objectI.GetRadius() + objectJ.GetRadius()) * 0.001; // Combined radii in kilometres

	if (relativeGravity)
	{
		escapeVelocity2 = 2 * (objectI.GetMass() + objectJ.GetMass()) * GravitationalConstant / boundingRadii;
		gravitationalPerturbation = (1 + escapeVelocity2 / relativeVelocity.vectorNorm2());
	}
	else
		gravitationalPerturbation = 1;

	return gravitationalPerturbation * Pi * boundingRadii * boundingRadii;
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
