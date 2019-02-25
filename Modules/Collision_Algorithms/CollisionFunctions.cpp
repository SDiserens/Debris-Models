
#include "stdafx.h"
#include "Collisions.h"



double CollisionAlgorithm::GetElapsedTime()
{
	return elapsedTime;
}

vector<CollisionPair> CollisionAlgorithm::CreatePairList(DebrisPopulation & population)
{
	vector<CollisionPair> pairList;
	int i, j;
	// For each object in population -
	for (i=0; i < population.population.size() -1 ; i++)
	{
		// For each subsequent object
		for (j = i + 1; j < population.population.size(); j++)
		{
			/// Add pair to list
			pairList.push_back(CollisionPair(population.population[i], population.population[j]));
		}
	}

	return pairList;
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

double CollisionAlgorithm::CalculateClosestApproach(CollisionPair objectPair)
{
	// TODO -  Set objects to closest position and return seperation
	return 0.0;
}


void CollisionAlgorithm::SwitchGravityComponent()
{
	relativeGravity = !relativeGravity;
}

CollisionPair::CollisionPair(DebrisObject & objectI, DebrisObject & objectJ)
{
	primary = objectI;
	secondary = objectJ;
	primaryID = objectI.GetID();
	secondaryID = objectJ.GetID();
	CalculateRelativeInclination();
	CalculateArgumenstOfIntersection();
}

CollisionPair::CollisionPair(long IDI, long IDJ)
{
	primaryID = IDI;
	secondaryID = IDJ;
}

double CollisionPair::GetRelativeInclination()
{
	return relativeInclination;
}

vector<pair<double, double>> CollisionPair::CalculateAngularWindowPrimary(double distance)
{
	return CalculateAngularWindow(primary, distance);
}

vector<pair<double, double>> CollisionPair::CalculateAngularWindowSecondary(double distance)
{
	return CalculateAngularWindow(secondary, distance);
}

vector3D CollisionPair::GetPrimaryPositionAtTime(double timeFromEpoch)
{
	//TODO - position at time
	return vector3D();
}

vector3D CollisionPair::GetPrimaryVelocityAtTime(double timeFromEpoch)
{
	//TODO - velcoity at time
	return vector3D();
}

vector3D CollisionPair::GetSecondaryPositionAtTime(double timeFromEpoch)
{
	//TODO -  position2 at time
	return vector3D();
}

vector3D CollisionPair::GetSecondaryVelocityAtTime(double timeFromEpoch)
{
	//TODO - velocity2 at time
	return vector3D();
}

double CollisionPair::CalculateMinimumSeparation()
{
	double trueAnomalyP, trueAnomalyS, circularAnomalyP, circularAnomalyS, cosRI, seperation, altSeperation;
	trueAnomalyP = deltaPrimary - primary.GetElements().argPerigee;
	trueAnomalyS = deltaSecondary - secondary.GetElements().argPerigee;

	// Find closest approach for elliptical orbits
	if (primary.GetElements().eccentricity != 0 || secondary.GetElements().eccentricity != 0)
	{
		int it = 0;
		double F, G, FdfP, FdfS, GdfP, GdfS;
		double tempAnomalyP, tempAnomalyS;
		double k, h = 1.0;

		circularAnomalyP = tempAnomalyP = trueAnomalyP;
		circularAnomalyS = tempAnomalyS = trueAnomalyS;
		cosRI = cos(relativeInclination);

		//Todo - Min Sep newton method
		while ((abs(h) >= NEWTONTOLERANCE || abs(k) >= NEWTONTOLERANCE) && (it < NEWTONMAXITERATIONS))
		{
			F = F; //TODO
			G = G; //TODO
			FdfP = 0;
			FdfS = 0;
			GdfP = 0;
			GdfS = 0;


			h = (F * GdfS - G * FdfS) / (FdfS*GdfP - FdfP*GdfS);
			k = (G * GdfP - F * FdfP) / (FdfS*GdfP - FdfP*GdfS);

			// Update values
			tempAnomalyP += h;
			tempAnomalyS += k;
			it++;
		}
		trueAnomalyP = tempAnomalyP;
		trueAnomalyS = tempAnomalyS;
	}

	primary.SetTrueAnomaly(trueAnomalyP);
	secondary.SetTrueAnomaly(trueAnomalyS);
	seperation = primary.GetPosition().CalculateRelativeVector(secondary.GetPosition()).vectorNorm();

	trueAnomalyP = fmod(trueAnomalyP + Pi, Tau);
	trueAnomalyS = fmod(trueAnomalyS + Pi, Tau);

	primary.SetTrueAnomaly(trueAnomalyP);
	secondary.SetTrueAnomaly(trueAnomalyS);
	altSeperation = primary.GetPosition().CalculateRelativeVector(secondary.GetPosition()).vectorNorm();

	return min(seperation, altSeperation);
}

double CollisionPair::GetBoundingRadii()
{
	return (primary.GetRadius() + secondary.GetRadius()) * 0.001; // Combined radii in kilometres;
}

double CollisionPair::CalculateSeparationAtTime(double timeFromEpoch)
{
	//TODO - closest approach distance
	return 0.0;
}

void CollisionPair::CalculateRelativeInclination()
{
	//TODO - Calculate relative inclination
	/*
	sin IR = |cross(hP, hC)|

	where r hP is the normal to the orbit plane of the primary object
	*/
	relativeInclination = 0;
}


void CollisionPair::CalculateArgumenstOfIntersection()
{
	// TODO Arguments of intersection
	deltaPrimary = 0;
	deltaSecondary = 0;
}

void CollisionPair::CalculateArgumenstOfIntersectionCoplanar()
{
	// TODO Coplanar Arguments of intersection
	deltaPrimary = 0;
	deltaSecondary = 0;
}

vector<pair<double, double>> CollisionPair::CalculateAngularWindow(DebrisObject & object, double distance)
{
	vector<pair<double, double>> timeWindows;
	//TODO - Calculate Angular Windows

	//TODO - check for singular case
	return timeWindows;
}
