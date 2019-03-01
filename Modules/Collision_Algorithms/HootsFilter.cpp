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
	interval = min(periodP, periodS) / 15;  // recommended fraction

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
	//TODO - closest approach time
	return 0.0;
}

double HootsFilter::CalculateFirstDerivateSeparation(CollisionPair objectPair, double candidateTime)
{
	//TODO - 1st derivative seperation
	return 0.0;
}

double HootsFilter::CalculateSecondDerivativeSeparation(CollisionPair objectPair, double candidateTime)
{
	// TODO - 2nd derivative seperation
	return 0.0;
}


