// OrbitTrace.cpp : contains the implementation of the Orbit Trace collision algorithm.
//

#include "stdafx.h"
#include "OrbitTrace.h"


OrbitTrace::OrbitTrace(bool probabilities, double threshold, int moid)
{
	outputProbabilities = probabilities;
	pAThreshold = threshold;
	MOIDtype = moid;

	if (MOIDtype == 1){
		detect_suitable_options(max_root_error, min_root_error, max_anom_error);
	}
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
	double tempProbability, collisionRate, altitude, mass, correction;
	list<CollisionPair> pairList;
	list<CollisionPair>::iterator listEnd;
	pair<long, long> pairID;
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
			if (!filters)
				objectPair.collision = true;
			else if (HeadOnFilter(objectPair) || !SynchronizedFilter(objectPair) || ProximityFilter(objectPair))
				objectPair.collision = true;
		}
		else
		{
			// Calculate intersections for non coplanar
			objectPair.CalculateArgumenstOfIntersection();
			if (!filters)
				objectPair.collision = true;
			else if (!SynchronizedFilter(objectPair) || ProximityFilter(objectPair))
				objectPair.collision = true;

			if (objectPair.collision) {
				if (objectPair.CalculateLowerBoundSeparation() > max(pAThreshold, objectPair.boundingRadii))
				{
					objectPair.collision = false;
				}
			}
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


	concurrency::parallel_for_each(pairList.begin(), pairList.end(), [&](CollisionPair& objectPair){
	//for (CollisionPair objectPair : pairList) {

		if (objectPair.constellation < 0)
			correction = 1;
		else
			correction = newSpaceCorrection;
		tempProbability = timeStep * objectPair.probability * correction;
		pairID = make_pair(objectPair.primaryID, objectPair.secondaryID);

		altitude = objectPair.primaryElements.GetRadialPosition();
		mass = objectPair.primaryMass + objectPair.secondaryMass;
		Event tempEvent(population.GetEpoch(), pairID.first, pairID.second, objectPair.GetRelativeVelocity(), mass, altitude, objectPair.minSeperation, tempProbability);
		tempEvent.SetCollisionAnomalies(objectPair.approachAnomalyP, objectPair.approachAnomalyS);

		// Store Collisions 
		mtx.lock();
		newCollisionList.push_back(tempEvent);
		mtx.unlock();

	}
	);
	
	elapsedTime += timeStep;

}


void OrbitTrace::MainCollision(DebrisPopulation& population, double timestep)
{
	double tempProbability, collisionRate, altitude, mass, lowerbound, correction;
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
			if (!filters)
				collision = true;
			else if (HeadOnFilter(objectPair) || !SynchronizedFilter(objectPair) || ProximityFilter(objectPair))
				collision = true;
		}
		else
		{
			// Calculate intersections for non coplanar
			objectPair.CalculateArgumenstOfIntersection();
			if (!filters)
				collision = true;
			else if (!SynchronizedFilter(objectPair) || ProximityFilter(objectPair))
				collision = true;

			if (objectPair.collision) {
				lowerbound = objectPair.CalculateLowerBoundSeparation();
				if (lowerbound > max(pAThreshold, objectPair.boundingRadii))
				{
					objectPair.collision = false;
				}
			}
		}

		if (collision)
		{
			collisionRate = CollisionRate(objectPair);
			if (objectPair.constellation < 0)
				correction = 1;
			else
				correction = newSpaceCorrection;
			tempProbability = timeStep * collisionRate * correction;

			if (tempProbability > 0) {
				pairID = make_pair(objectPair.primaryID, objectPair.secondaryID);

				altitude = objectPair.primaryElements.GetRadialPosition();
				mass = objectPair.primaryMass + objectPair.secondaryMass;
				Event tempEvent(population.GetEpoch(), pairID.first, pairID.second, objectPair.GetRelativeVelocity(), mass, altitude, objectPair.minSeperation, tempProbability);
				tempEvent.SetCollisionAnomalies(objectPair.approachAnomalyP, objectPair.approachAnomalyS);
				// Store Collisions 
				//collisionList.push_back(tempEvent);
				newCollisionList.push_back(tempEvent);
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
	double collisionRate, collisionRate2, boundingRadii, minSeperation, scaling, relativeVelocity, threshold, escapeVelocity2, gravitationalPerturbation;
	vector3D velocityI, velocityJ;

	//TODO - Quick filter on possible separation
	switch (MOIDtype) {
	case 0: 
		minSeperation = objectPair.CalculateMinimumSeparation();
		break;
	case 1: 
		minSeperation = objectPair.CalculateMinimumSeparation_DL(max_root_error, min_root_error, max_anom_error);
		break;
	case 2: 
		minSeperation = objectPair.CalculateMinimumSeparation_MOID();
		break;
	}
	//minSeperation = objectPair.CalculateMinimumSeparation();

	boundingRadii = objectPair.GetBoundingRadii();
	threshold = max(pAThreshold, boundingRadii);
	//sinAngle = velocityI.VectorCrossProduct(velocityJ).vectorNorm() / (velocityI.vectorNorm() * velocityJ.vectorNorm());
	scaling = 1;
	if (boundingRadii < pAThreshold) {
		scaling = boundingRadii / pAThreshold;
		scaling = scaling * scaling;
	}

	if (relativeGravity)
	{
		escapeVelocity2 = 2 * (objectPair.primaryMass + objectPair.secondaryMass) * GravitationalConstant / boundingRadii;
	}

	// OT collision rate
	if (objectPair.minSeperation < threshold && objectPair.minSeperation >= 0)
	{	
		objectPair.primaryElements.SetTrueAnomaly(objectPair.approachAnomalyP);
		objectPair.secondaryElements.SetTrueAnomaly(objectPair.approachAnomalyS);
		velocityI = objectPair.primaryElements.GetVelocity();
		velocityJ = objectPair.secondaryElements.GetVelocity();
		relativeVelocity = objectPair.relativeVelocity = velocityI.CalculateRelativeVector(velocityJ).vectorNorm();
		if (relativeGravity)
		{
			gravitationalPerturbation = (1 + escapeVelocity2 / (relativeVelocity * relativeVelocity));
		}
		else
			gravitationalPerturbation = 1;

		collisionRate = gravitationalPerturbation * Pi * threshold * relativeVelocity /
			(2 * velocityI.VectorCrossProduct(velocityJ).vectorNorm()  * objectPair.primaryElements.CalculatePeriod() * objectPair.secondaryElements.CalculatePeriod());
	}
	else
		collisionRate = 0;

	if (objectPair.minSeperation2 < threshold && objectPair.minSeperation2 >= 0)
	{

		objectPair.primaryElements.SetTrueAnomaly(objectPair.approachAnomalyP2);
		objectPair.secondaryElements.SetTrueAnomaly(objectPair.approachAnomalyS2);
		velocityI = objectPair.primaryElements.GetVelocity();
		velocityJ = objectPair.secondaryElements.GetVelocity();
		relativeVelocity = objectPair.relativeVelocity2 = velocityI.CalculateRelativeVector(velocityJ).vectorNorm();
		if (relativeGravity)
		{
			gravitationalPerturbation = (1 + escapeVelocity2 / (relativeVelocity * relativeVelocity));
		}
		else
			gravitationalPerturbation = 1;

		collisionRate2 = gravitationalPerturbation * Pi * threshold * relativeVelocity /
			(2 * velocityI.VectorCrossProduct(velocityJ).vectorNorm()  * objectPair.primaryElements.CalculatePeriod() * objectPair.secondaryElements.CalculatePeriod());

		if (abs(objectPair.approachAnomalyP2 - objectPair.approachAnomalyP) * objectPair.primaryElements.semiMajorAxis > boundingRadii)
			collisionRate += collisionRate2;
		else
			collisionRate = max(collisionRate, collisionRate2);
	}
	else
		collisionRate2 = 0;

	return scaling * collisionRate;
}


bool OrbitTrace::CoplanarFilter(CollisionPair& objectPair)
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

void OrbitTrace::SwitchFilters()
{
	filters = !filters;
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

