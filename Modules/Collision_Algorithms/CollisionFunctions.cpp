#include "stdafx.h"
#include "Collisions.h"

double CollisionAlgorithm::GetElapsedTime()
{
	return elapsedTime;
}

vector<double> CollisionAlgorithm::GetCollisionVerbose()
{
	return GetCollisionProbabilities();
}

vector<double> CollisionAlgorithm::GetNewCollisionVerbose()
{
	return GetNewCollisionProbabilities();
}

vector<double> CollisionAlgorithm::GetNewCollisionAltitudes()
{
	vector<double> newList(newCollisionAltitudes);
	newCollisionAltitudes.clear();
	return newList;
}

list<CollisionPair> CollisionAlgorithm::CreatePairList(DebrisPopulation & population)
{
	list<CollisionPair> pairList;

	// For each object in population -
	for (auto it=population.population.begin(); it!= population.population.end(); it++)
	{
		// For each subsequent object
		auto jt = it;
		for (++jt; jt != population.population.end(); ++jt)
		{
			/// Add pair to list
			//DebrisObject& primaryObject(population.Ge), secondaryObject;
			CollisionPair pair(it->second, jt->second);
			if (PerigeeApogeeTest(pair))
				pairList.push_back(pair);
			else
				pair.~CollisionPair();
		}
	}

	return pairList;
}


list<CollisionPair> CollisionAlgorithm::CreatePairList_P(DebrisPopulation & population)
{
	list<CollisionPair> pairList;
	// For each object in population - parallelised
	
	mutex mtx;
	concurrency::parallel_for_each(population.population.begin(), population.population.end(), [&](auto& it) {
		auto jt = population.population.find(it.first);
		for (++jt; jt != population.population.end(); ++jt)
		{
			/// Add pair to list
			//DebrisObject& primaryObject(population.Ge), secondaryObject;
			CollisionPair pair(it.second, jt->second);
			if (PerigeeApogeeTest(pair)) {
				mtx.lock();
				pairList.push_back(pair);
				mtx.unlock();
			}
		else
			pair.~CollisionPair();
		}
		});
	return pairList;
}



bool CollisionAlgorithm::PerigeeApogeeTest(CollisionPair& objectPair)
{
	double maxPerigee, minApogee;
	// Perigee Apogee Test
	maxPerigee = max(objectPair.primary.GetPerigee(), objectPair.secondary.GetPerigee());
	minApogee = min(objectPair.primary.GetApogee(), objectPair.secondary.GetApogee());

	return (maxPerigee - minApogee) <= pAThreshold;
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

vector<Event> CollisionAlgorithm::GetCollisionList()
{
	return collisionList;
}

vector<double> CollisionAlgorithm::GetCollisionProbabilities()
{
	return collisionProbabilities;
}

vector<Event> CollisionAlgorithm::GetNewCollisionList()
{
	vector<Event> newList(newCollisionList);
	newCollisionList.clear();
	return newList;
}

vector<double> CollisionAlgorithm::GetNewCollisionProbabilities()
{
	vector<double> newList(newCollisionProbabilities);
	newCollisionProbabilities.clear();
	return newList;
}

bool CollisionAlgorithm::DetermineCollisionAvoidance(double avoidanceProbability)
{
	return randomNumber() < avoidanceProbability;
}

bool CollisionAlgorithm::UseGPU()
{
	return GPU;
}

bool CollisionAlgorithm::UseParallel()
{
	return parallel;
}

bool CollisionAlgorithm::DetermineCollision(double collisionProbability)
{
	return randomNumber() < collisionProbability;
}

/*
double CollisionAlgorithm::CalculateClosestApproach(CollisionPair objectPair)
{
	//  Set objects to position at close approach time and return seperation
	return objectPair.CalculateSeparationAtTime();
}
*/

void CollisionAlgorithm::MainCollision_P(DebrisPopulation & population, double timeStep)
{
	MainCollision(population, timeStep);
}

void CollisionAlgorithm::MainCollision_GPU(DebrisPopulation & population, double timeStep)
{
	MainCollision(population, timeStep);
}

void CollisionAlgorithm::SwitchGravityComponent()
{
	relativeGravity = !relativeGravity;
}

void CollisionAlgorithm::SwitchParallelGPU()
{
	GPU = !GPU;
}

void CollisionAlgorithm::SwitchParallelCPU()
{
	parallel = !parallel;
}
