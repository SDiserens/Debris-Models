#include "stdafx.h"
#include "HootsFilter.h"


HootsFilter::HootsFilter(double init_conjThreshold, double init_collThreshold)
{
	conjunctionThreshold = init_conjThreshold;
	collisionThreshold = init_collThreshold;
}

void HootsFilter::MainCollision(DebrisPopulation & population, double timeStep)
{
}

double HootsFilter::CollisionRate(DebrisObject & objectI, DebrisObject & objectJ)
{
	return 0.0;
}

bool HootsFilter::PerigeeApogeeTest(CollisionPair objectPair)
{
	double maxApogee, minPerigee;
	// Perigee Apogee Test
	minPerigee = min(objectPair.primary.GetPerigee(), objectPair.secondary.GetPerigee());
	maxApogee = max(objectPair.primary.GetApogee(), objectPair.secondary.GetApogee());

	return (maxApogee - minPerigee) <= conjunctionThreshold;
}

bool HootsFilter::GeometricFilter(CollisionPair objectPair)
{
	return objectPair.CalculateMinimumSeparation() <= conjunctionThreshold;
}

bool HootsFilter::TimeFilter(CollisionPair objectPair, double timeStep)
{
	// TODO - Time Filter

	return false;
}

bool HootsFilter::CoplanarFilter(CollisionPair objectPair, double timeStep)
{
	// Coplanar Filter
	double candidateTime, time, rate, previousTime, previousRate, periodP, periodS, interval;
	vector<double> candidateTimeList;

	time = rate = previousTime = 0;
	periodP = objectPair.primary.GetPeriod();
	periodS = objectPair.secondary.GetPeriod();
	interval = min(periodP, periodS) / 5;  // recommended fraction

	while (time < timeStep)
	{
		previousRate = rate;
		rate = CalculateFirstDerivateSeparation(objectPair, time);
		if (rate == 0 && previousRate < 0)
			candidateTimeList.push_back(time);
		else if (previousRate < 0 && rate > 0)
		{
			candidateTime = previousTime + interval * (previousRate / (previousRate - rate));
			candidateTimeList.push_back(candidateTime);
		}
		previousTime = time;
		time += interval;

	}
	return false;
}

vector<double> HootsFilter::DetermineCollisionTimes(CollisionPair objectPair, vector<double> candidateTimeList)
{
	// TODO - Collision Times
	return vector<double>();
}

vector<pair<double, double>> HootsFilter::CalculateTimeWindows(pair<double, double> window, double period, double timestep)
{
	//TODO - Time windows
	return vector<pair<double, double>>();
}

double HootsFilter::CalculateClosestApproachTime(CollisionPair objectPair, double candidateTime)
{
	int it = 0;
	double approachTime, R, Rdot, h = 1.0;
	// Closest approach time
	approachTime = candidateTime;

	while ((abs(h) >= NEWTONTOLERANCE) && (it < NEWTONMAXITERATIONS))
	{
		R = CalculateFirstDerivateSeparation(objectPair, approachTime);
		Rdot = CalculateSecondDerivativeSeparation(objectPair, approachTime);
		h = R / Rdot;
		approachTime -= h;
		it++;
	}
	return approachTime;
}

double HootsFilter::CalculateFirstDerivateSeparation(CollisionPair objectPair, double candidateTime)
{
	//1st derivative seperation
	double rDot;
	vector3D positionP, positionS, velocityP, velocityS;

	positionP = objectPair.GetPrimaryPositionAtTime(candidateTime);
	positionS = objectPair.GetSecondaryPositionAtTime(candidateTime);
	velocityP = objectPair.GetPrimaryVelocityAtTime(candidateTime);
	velocityS = objectPair.GetSecondaryVelocityAtTime(candidateTime);

	rDot = positionP.VectorDotProduct(velocityP) + positionS.VectorDotProduct(velocityS) - velocityP.VectorDotProduct(positionS) - positionP.VectorDotProduct(velocityS);

	return rDot;
}

double HootsFilter::CalculateSecondDerivativeSeparation(CollisionPair objectPair, double candidateTime)
{
	// 2nd derivative seperation
	double rDotDot;
	vector3D positionP, positionS, velocityP, velocityS, accelerationP, accelerationS;

	positionP = objectPair.GetPrimaryPositionAtTime(candidateTime);
	positionS = objectPair.GetSecondaryPositionAtTime(candidateTime);
	velocityP = objectPair.GetPrimaryVelocityAtTime(candidateTime);
	velocityS = objectPair.GetSecondaryVelocityAtTime(candidateTime);

	accelerationP = CalculateAcceleration(positionP);
	accelerationS = CalculateAcceleration(positionS);

	rDotDot = velocityP.vectorNorm2() + positionP.vectorNorm() * accelerationP.vectorNorm() + velocityS.vectorNorm2() + positionS.vectorNorm() * accelerationS.vectorNorm()
		- accelerationP.VectorDotProduct(positionS) - 2 * velocityP.VectorDotProduct(velocityS) - velocityP.VectorDotProduct(accelerationS);

	return rDotDot;
}


