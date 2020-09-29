#include "stdafx.h"
#include "Collisions.h"

double CollisionAlgorithm::GetElapsedTime()
{
	return elapsedTime;
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
	size_t n = population.GetPopulationSize();
	size_t N = n * (n - 1) / 2.0;
	vector<long> keys;
	list<CollisionPair> pairList(N);
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

	/*

	for_each(population.population.begin(), population.population.end(), [&](auto& it) {
		keys.push_back(it.first);
	});

	concurrency::parallel_for_each(size_t(0), N, [&](size_t i) {
		size_t z = n - 1;
		size_t x = 1;
		while (i > z) {
			i -= z;
			--z;
			++x;
		}
		size_t y = x + i;
		pairList[i] = CollisionPair(population.GetObject(keys.at(--x)), population.GetObject(keys.at(--y)));
	});
	*/
	return pairList;
}



bool CollisionAlgorithm::PerigeeApogeeTest(CollisionPair& objectPair)
{
	double maxPerigee, minApogee;
	// Perigee Apogee Test
	maxPerigee = max(objectPair.primaryElements.GetPerigee(), objectPair.secondaryElements.GetPerigee());
	minApogee = min(objectPair.primaryElements.GetApogee(), objectPair.secondaryElements.GetApogee());

	return (maxPerigee - minApogee) <= max(pAThreshold, objectPair.GetBoundingRadii());
}

double CollisionAlgorithm::CollisionCrossSection(CollisionPair& objectPair)
{
	double boundingRadii, escapeVelocity2, gravitationalPerturbation;
	vector3D velocityI = objectPair.primaryElements.GetVelocity();
	vector3D velocityJ = objectPair.secondaryElements.GetVelocity();

	vector3D relativeVelocity = velocityI.CalculateRelativeVector(velocityJ);
	boundingRadii = objectPair.GetBoundingRadii(); // Combined radii in kilometres

	if (relativeGravity)
	{
		escapeVelocity2 = 2 * (objectPair.primaryMass + objectPair.secondaryMass) * GravitationalConstant / boundingRadii;
		gravitationalPerturbation = (1 + escapeVelocity2 / relativeVelocity.vectorNorm2());
	}
	else
		gravitationalPerturbation = 1;

	return gravitationalPerturbation * Pi * boundingRadii * boundingRadii;
}

vector<Event> CollisionAlgorithm::GetNewCollisionList()
{
	vector<Event> newList(newCollisionList);
	newCollisionList.clear();
	newCollisionList.shrink_to_fit();
	return newList;
}

bool CollisionAlgorithm::DetermineCollisionAvoidance(double avoidanceProbability)
{
	return randomNumber() < avoidanceProbability;
}

bool CollisionAlgorithm::CheckValidCollision(DebrisObject target, DebrisObject projectile)
{
	bool valid = true;
	//logic for invalid collision
	if (projectile.GetConstellationID() == target.GetConstellationID() && projectile.IsActive() && target.IsActive()) 
	{
		valid = false;
	}
	else if(abs(target.GetPeriod() - projectile.GetPeriod())/ target.GetPeriod() < 0.01)
	{
		vector3D hP = target.GetElements().GetNormalVector();
		vector3D hS = projectile.GetElements().GetNormalVector();
		double k = hP.VectorCrossProduct(hS).vectorNorm();

		double relativeInclination = asin(k);
		double combinedSemiMajorAxis = target.GetElements().semiMajorAxis + projectile.GetElements().semiMajorAxis;
		bool coplanar = relativeInclination <= (2 * asin((target.GetRadius() + projectile.GetRadius()) / (1000 * combinedSemiMajorAxis)));
		if (coplanar) {
			valid = false;
		}
	}
	return valid;
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

void CollisionAlgorithm::SetNewSpaceParameters(double correction)
{
	newSpace = true;
	newSpaceCorrection = correction;

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
