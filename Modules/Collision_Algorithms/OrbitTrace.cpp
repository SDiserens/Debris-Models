// OrbitTrace.cpp : contains the implementation of the Orbit Trace collision algorithm.
//

#include "stdafx.h"
#include "OrbitTrace.h"


OrbitTrace::OrbitTrace(bool probabilities, double threshold)
{
	outputProbabilities = probabilities;
	pAThreshold = threshold;
}

OrbitTrace::OrbitTrace(bool probabilities, double threshold, int moid)
{
	outputProbabilities = probabilities;
	pAThreshold = threshold;
	MOIDtype = moid;
}


void OrbitTrace::SetThreshold(double threshold)
{
	pAThreshold = threshold;
}

void OrbitTrace::SetMOID(int moid)
{
	cout << "Moid set to " << moid << "\n";
	MOIDtype = moid;
}

void OrbitTrace::MainCollision_P(DebrisPopulation& population, double timestep)
{
	double tempProbability, collisionRate, altitude, mass;
	list<CollisionPair> pairList;
	list<CollisionPair>::iterator listEnd;
	pair<long, long> pairID;
	bool collision;
	mutex mtx;
	// Filter Cube List
	pairList = CreatePairList_P(population);
	timeStep = timestep;

	concurrency::parallel_for_each(pairList.begin(), pairList.end(), [&](CollisionPair& objectPair)
	{
		objectPair.collision = false;

		objectPair.CalculateRelativeInclination();

		if (CoplanarFilter(objectPair))
		{
			// Calculate orbit intersections for coplanar
			objectPair.CalculateArgumenstOfIntersectionCoplanar();
			if (HeadOnFilter(objectPair) || !SynchronizedFilter(objectPair) || ProximityFilter(objectPair))
				objectPair.collision = true;
		}
		else
		{
			// Calculate intersections for non coplanar
			objectPair.CalculateArgumenstOfIntersection();
			if (!SynchronizedFilter(objectPair) || ProximityFilter(objectPair))
				objectPair.collision = true;
		}

		if (objectPair.collision)
		{
			collisionRate = CollisionRate(objectPair);
			objectPair.probability = collisionRate;
		}
		else
			objectPair.probability = 0;
	});
	listEnd = remove_if(pairList.begin(), pairList.end(), [&](CollisionPair& objectPair) {
		return (objectPair.probability == 0);
	});
	pairList.erase(listEnd, pairList.end());

	for (CollisionPair objectPair : pairList) {

		tempProbability = timeStep * objectPair.probability;
		pairID = make_pair(objectPair.primaryID, objectPair.secondaryID);

		altitude = objectPair.primaryElements.GetRadialPosition();
		mass = objectPair.primaryMass + objectPair.secondaryMass;
		Event tempEvent(population.GetEpoch(), pairID.first, pairID.second, objectPair.GetRelativeVelocity(), mass, altitude, objectPair.GetMinSeparation());
		tempEvent.SetCollisionAnomalies(objectPair.approachAnomalyP, objectPair.approachAnomalyS);
		//	-- Determine if collision occurs through MC (random number generation)
		if (outputProbabilities && tempProbability > 0)
		{
			//	-- Store collision probability
			newCollisionProbabilities.push_back(tempProbability);
			newCollisionList.push_back(tempEvent);
		}
		else
		{
			if (DetermineCollision(tempProbability))
			{
				// Store Collisions 
				newCollisionList.push_back(tempEvent);
			}
		}

	}
	
	elapsedTime += timeStep;

}


void OrbitTrace::MainCollision(DebrisPopulation& population, double timestep)
{
	double tempProbability, collisionRate, altitude, mass;
	list<CollisionPair> pairList;
	pair<long, long> pairID;
	bool collision;

	// Filter Cube List
	pairList = CreatePairList(population);
	timeStep = timestep;
	
	for (CollisionPair& objectPair : pairList)
	{
		collision = false;

		objectPair.CalculateRelativeInclination();

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

			altitude = objectPair.primaryElements.GetRadialPosition();
			mass = objectPair.primaryMass + objectPair.secondaryMass;
			Event tempEvent(population.GetEpoch(), pairID.first, pairID.second, objectPair.GetRelativeVelocity(), mass, altitude, objectPair.GetMinSeparation());
			tempEvent.SetCollisionAnomalies(objectPair.approachAnomalyP, objectPair.approachAnomalyS);
			//	-- Determine if collision occurs through MC (random number generation)
			if (outputProbabilities && tempProbability>0)
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
					//collisionList.push_back(tempEvent);
					newCollisionList.push_back(tempEvent);
				}
			}
		}
	}
	elapsedTime += timeStep;

}

/*
double OrbitTrace::CollisionRate(CollisionPair &objectPair)
{
	double collisionRate, boundingRadii, minSeperation, sinAngle;//  , sinAngleV2;
	vector3D velocityI, velocityJ, relativeVelocity;

	minSeperation = objectPair.CalculateMinimumSeparation();

	velocityI = objectPair.primary.GetVelocity();
	velocityJ = objectPair.secondary.GetVelocity();

	relativeVelocity = velocityI.CalculateRelativeVector(velocityJ);
	boundingRadii = max(pAThreshold, objectPair.GetBoundingRadii());

	sinAngle = velocityI.VectorCrossProduct(velocityJ).vectorNorm() / (velocityI.vectorNorm() * velocityJ.vectorNorm());

	// OT collision rate
	if (boundingRadii > minSeperation)
		collisionRate = 2 * sqrt(boundingRadii*boundingRadii - minSeperation * minSeperation) / 
			(sinAngle * relativeVelocity.vectorNorm() * objectPair.primary.GetPeriod() * objectPair.secondary.GetPeriod());
	else
		collisionRate = 0;

	return collisionRate;
}
*/

double OrbitTrace::CollisionRate(CollisionPair &objectPair)
{
	double collisionRate, boundingRadii, minSeperation, relativeVelocity;
	vector3D velocityI, velocityJ;

	//TODO - Quick filter on possible separation
	switch (MOIDtype) {
	case 0: 
		minSeperation = objectPair.CalculateMinimumSeparation();
		break;
	case 1: 
		minSeperation = objectPair.CalculateMinimumSeparation_DL();
		break;
	case 2: 
		minSeperation = objectPair.CalculateMinimumSeparation_MOID();
		break;
	}
	

	velocityI = objectPair.primaryElements.GetVelocity();
	velocityJ = objectPair.secondaryElements.GetVelocity();

	relativeVelocity = velocityI.CalculateRelativeVector(velocityJ).vectorNorm();
	boundingRadii = max(pAThreshold, objectPair.GetBoundingRadii());
	objectPair.SetRelativeVelocity(relativeVelocity);
	//sinAngle = velocityI.VectorCrossProduct(velocityJ).vectorNorm() / (velocityI.vectorNorm() * velocityJ.vectorNorm());

	// OT collision rate
	if (boundingRadii > minSeperation)
		collisionRate = Pi * boundingRadii * relativeVelocity /
						(2 * velocityI.VectorCrossProduct(velocityJ).vectorNorm()  * objectPair.primaryElements.CalculatePeriod() * objectPair.secondaryElements.CalculatePeriod());
	else
		collisionRate = 0;

	return collisionRate;
}


bool OrbitTrace::CoplanarFilter(CollisionPair objectPair)
{
	// Coplanar filter
	double combinedSemiMajorAxis = objectPair.primaryElements.semiMajorAxis + objectPair.secondaryElements.semiMajorAxis;
	bool coplanar = objectPair.GetRelativeInclination() <= (2 * asin(objectPair.GetBoundingRadii() / combinedSemiMajorAxis));
	objectPair.coplanar = coplanar;
	return coplanar;
}

bool OrbitTrace::HeadOnFilter(CollisionPair objectPair)
{
	bool headOn = false;
	double deltaW;
	double eLimitP = objectPair.GetBoundingRadii() / objectPair.primaryElements.semiMajorAxis;
	double eLimitS = objectPair.GetBoundingRadii() / objectPair.secondaryElements.semiMajorAxis;
	// OT Head on filter
	if ((objectPair.primaryElements.eccentricity <= eLimitP) && (objectPair.secondaryElements.eccentricity <= eLimitS))
			headOn = true;
	else
	{
		deltaW = abs(Pi - objectPair.primaryElements.argPerigee - objectPair.secondaryElements.argPerigee);
		if (deltaW <= 1)
			headOn = true;
		else if (Tau - deltaW <= 1)
			headOn = true;
	}
	
	return headOn;
}

bool OrbitTrace::SynchronizedFilter(CollisionPair objectPair)
{
	double meanMotionP, meanMotionS, driftAngle;
	// OT synch filter
	meanMotionP = Tau / objectPair.primaryElements.CalculatePeriod();
	meanMotionS = Tau / objectPair.secondaryElements.CalculatePeriod();

	driftAngle = abs(meanMotionP - meanMotionS) * timeStep;
	return (driftAngle >= Tau);
}

bool OrbitTrace::ProximityFilter(CollisionPair objectPair)
{
	//  OT  proximity filter
	double deltaMP, deltaMS, deltaMAngle, deltaMLinear, combinedSemiMajorAxis; 
	OrbitalAnomalies anomaliesP, anomaliesS;

	anomaliesP.SetTrueAnomaly(objectPair.approachAnomalyP);
	anomaliesS.SetTrueAnomaly(objectPair.approachAnomalyS);

	deltaMP = abs(anomaliesP.GetMeanAnomaly(objectPair.primaryElements.eccentricity) - objectPair.primaryElements.GetMeanAnomaly());
	deltaMS = abs(anomaliesS.GetMeanAnomaly(objectPair.secondaryElements.eccentricity) - objectPair.secondaryElements.GetMeanAnomaly());
	
	combinedSemiMajorAxis = (objectPair.primaryElements.semiMajorAxis + objectPair.secondaryElements.semiMajorAxis) / 2;
	deltaMAngle = abs(deltaMP - deltaMS);
	deltaMLinear = deltaMAngle * combinedSemiMajorAxis;

	return (deltaMLinear <= objectPair.GetBoundingRadii());
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

