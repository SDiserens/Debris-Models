// OrbitTrace.cpp : contains the implementation of the Orbit Trace collision algorithm.
//

#include "stdafx.h"
#include "OrbitTrace.h"

OrbitTrace::OrbitTrace(bool probabilities, double threshold)
{
	outputProbabilities = probabilities;
	pAThreshold = threshold;
}


void OrbitTrace::SetThreshold(double threshold)
{
	pAThreshold = threshold;
}

void OrbitTrace::MainCollision(DebrisPopulation& population, double timestep)
{
	double tempProbability, collisionRate, altitude;
	vector<CollisionPair> pairList;
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

			//	-- Determine if collision occurs through MC (random number generation)
			if (outputProbabilities && tempProbability>0)
			{
				altitude = objectPair.GetCollisionAltitude();
				//	-- Store collision probability
				//collisionProbabilities.push_back(tempProbability);
				//collisionList.push_back(collisionPair);
				newCollisionProbabilities.push_back(tempProbability);
				newCollisionList.push_back(pairID);
				newCollisionAltitudes.push_back(altitude);
			}
			else
			{
				if (DetermineCollision(tempProbability))
				{
					altitude = objectPair.GetCollisionAltitude();
					// Store Collisions 
					collisionList.push_back(pairID);
					newCollisionList.push_back(pairID);
					newCollisionAltitudes.push_back(altitude);
				}
			}
		}
	}
	elapsedTime += timeStep;

}

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


bool OrbitTrace::CoplanarFilter(CollisionPair objectPair)
{
	// Coplanar filter
	double combinedSemiMajorAxis = objectPair.primary.GetElements().semiMajorAxis + objectPair.secondary.GetElements().semiMajorAxis;
	bool coplanar = objectPair.GetRelativeInclination() <= (2 * asin(objectPair.GetBoundingRadii() / combinedSemiMajorAxis));
	objectPair.coplanar = coplanar;
	return coplanar;
}

bool OrbitTrace::HeadOnFilter(CollisionPair objectPair)
{
	bool headOn = false;
	double deltaW;
	double eLimitP = objectPair.GetBoundingRadii() / objectPair.primary.GetElements().semiMajorAxis;
	double eLimitS = objectPair.GetBoundingRadii() / objectPair.secondary.GetElements().semiMajorAxis;
	// OT Head on filter
	if ((objectPair.primary.GetElements().eccentricity <= eLimitP) && (objectPair.secondary.GetElements().eccentricity <= eLimitS))
			headOn = true;
	else
	{
		deltaW = abs(Pi - objectPair.primary.GetElements().argPerigee - objectPair.secondary.GetElements().argPerigee);
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
	meanMotionP = Tau / objectPair.primary.GetPeriod();
	meanMotionS = Tau / objectPair.secondary.GetPeriod();

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

	deltaMP = abs(anomaliesP.GetMeanAnomaly(objectPair.primary.GetElements().eccentricity) - objectPair.primary.GetElements().GetMeanAnomaly());
	deltaMS = abs(anomaliesS.GetMeanAnomaly(objectPair.secondary.GetElements().eccentricity) - objectPair.secondary.GetElements().GetMeanAnomaly());
	
	combinedSemiMajorAxis = (objectPair.primary.GetElements().semiMajorAxis + objectPair.secondary.GetElements().semiMajorAxis) / 2;
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

