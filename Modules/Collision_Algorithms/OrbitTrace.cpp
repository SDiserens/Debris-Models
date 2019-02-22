// OrbitTrace.cpp : contains the implementation of the Orbit Trace collision algorithm.
//

#include "stdafx.h"
#include "OrbitTrace.h"


void OrbitTrace::MainCollision(DebrisPopulation& population, double timeStep)
{
	double tempProbability, collisionRate;
	vector<CollisionPair> pairList;
	pair<long, long> pairID;
	bool collision;
	// Filter Cube List
	pairList = CreatePairList(population);

	for (CollisionPair& objectPair : pairList)
	{
		collision = false;

		if (CoplanarFilter(objectPair))
		{
			// Calculate orbit intersections for coplanar
			objectPair.CalculateArgumenstOfIntersectionCoplanar();
			if (HeadOnFilter(objectPair) || !SynchronizedFilter(objectPair) || ProximityFilter(objectPair))
				collision = true;
		}
		else
		{
			// Calculate intersections for non coplanar
			objectPair.CalculateArgumenstOfIntersection();
			if (!SynchronizedFilter(objectPair) || ProximityFilter(objectPair))
				collision = true;
		}

		if (collision)
		{
			collisionRate = CollisionRate(objectPair);
			tempProbability = timeStep * collisionRate;
			pairID = make_pair(objectPair.primaryID, objectPair.secondaryID);

			//	-- Determine if collision occurs through MC (random number generation)
			if (outputProbabilities)
			{
				//	-- Store collision probability
				//collisionProbabilities.push_back(tempProbability);
				//collisionList.push_back(collisionPair);
				newCollisionProbabilities.push_back(tempProbability);
				newCollisionList.push_back(pairID);
			}
			else
			{
				if (DetermineCollision(tempProbability))
				{
					// Store Collisions 
					collisionList.push_back(pairID);
					newCollisionList.push_back(pairID);
				}
			}
		}
	}
	elapsedTime += timeStep;

}

double OrbitTrace::CollisionRate(CollisionPair objectPair)
{
	double collisionRate, boundingRadii, minSeperation, sinAngle;
	vector3D velocityI, velocityJ, relativeVelocity;

	minSeperation = objectPair.CalculateMinimumSeparation();

	velocityI = objectPair.primary.GetVelocity();
	velocityJ = objectPair.secondary.GetVelocity();

	relativeVelocity = velocityI.CalculateRelativeVector(velocityJ);
	boundingRadii = objectPair.GetBoundingRadii();

	sinAngle = velocityI.VectorCrossProduct(velocityJ).vectorNorm() / (velocityI.vectorNorm() * velocityJ.vectorNorm());

	//TODO - OT collision rate
	collisionRate = 2 * sqrt(boundingRadii*boundingRadii - minSeperation * minSeperation) / 
		(sinAngle * relativeVelocity.vectorNorm() * objectPair.primary.GetPeriod() * objectPair.secondary.GetPeriod());

	return collisionRate;
}


bool OrbitTrace::CoplanarFilter(CollisionPair objectPair)
{
	// TODO -OT Coplanar filter
	return false;
}

bool OrbitTrace::HeadOnFilter(CollisionPair objectPair)
{
	// TODO - OT Head on filter
	return false;
}

bool OrbitTrace::SynchronizedFilter(CollisionPair objectPair)
{
	// TODO - OT synch filter
	return false;
}

bool OrbitTrace::ProximityFilter(CollisionPair objectPair)
{
	// TODO - OT  proximity filter
	return false;
}

/*
double OrbitTrace::CalculateSpatialDensity(DebrisObject object, double radius, double latitude)
{
	// Equation 21
	return 0.0;
}

double OrbitTrace::CalculateRadialSpatialDensity(DebrisObject object, double radius)
{
	// Equation 8A
	return 0.0;
}

double OrbitTrace::CalculateLatitudinalSpatialDensityRatio(DebrisObject object, double latitude)
{
	// Equations 13A
	return 0.0;
}

double OrbitTrace::CalculateVolumeElement(double radius, double latitude)
{
	// Equation 17
	return 0.0;
}
*/

