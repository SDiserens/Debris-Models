
#include "stdafx.h"
#include "Collisions.h"



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

CollisionPair::CollisionPair(DebrisObject & objectI, DebrisObject & objectJ)
{
	primary = objectI;
	secondary = objectJ;
	CalculateRelativeInclination();
	CalculateArgumenstOfIntersection();
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
	// TODO - Min seperation
	return 0.0;
}

double CollisionPair::CalculateSeparationAtTime(double timeFromEpoch)
{
	//TODO - closest approach distance
	return 0.0;
}

void CollisionPair::CalculateRelativeInclination()
{
	//TODO - Calculate relative inclination
	relativeInclination = 0;
}


void CollisionPair::CalculateArgumenstOfIntersection()
{
	// TODO Arguments of intersection
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
