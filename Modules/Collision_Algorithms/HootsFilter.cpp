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

vector<double> HootsFilter::TimeFilter(CollisionPair objectPair, double timeStep)
{
	double time, eP, eS, periodP, periodS, candidateTime;
	vector<double> angularWindowPrimary, angularWindowSecondary, timeWindowPrimary, timeWindowSecondary, timeList;
	vector<pair<double, double>> timeWindowsP, timeWindowsS;
	// Time Filter
	angularWindowPrimary = objectPair.CalculateAngularWindowPrimary(conjunctionThreshold);
	if (angularWindowPrimary[0] < 0)
		return angularWindowPrimary;

	angularWindowSecondary = objectPair.CalculateAngularWindowSecondary(conjunctionThreshold);
	if (angularWindowSecondary[0] < 0)
		return angularWindowSecondary;

	// Convert to mean anomalies
	eP = objectPair.primary.GetElements().eccentricity;
	periodP = objectPair.primary.GetPeriod();
	for (double angle : angularWindowPrimary)
	{
		angle = objectPair.primary.GetElements().anomalies.TrueToMeanAnomaly(angle, eP);
		time = periodP * (angle - objectPair.primary.GetEpochAnomaly()) / Tau;
		timeWindowPrimary.push_back(time);
	}
	if (timeWindowPrimary[3] < timeWindowPrimary[2])
		timeWindowPrimary[3] += periodP;

	eS = objectPair.secondary.GetElements().eccentricity;
	periodS = objectPair.secondary.GetPeriod();
	for (double angle : angularWindowSecondary)
	{
		angle = objectPair.secondary.GetElements().anomalies.TrueToMeanAnomaly(angle, eP);
		time = periodP * (angle - objectPair.secondary.GetEpochAnomaly()) / Tau;
		timeWindowSecondary.push_back(time);
	}
	if (timeWindowSecondary[3] < timeWindowSecondary[2])
		timeWindowSecondary[3] += periodS;

	// When calling time-windows function need to do 4 times, once for each window for each object
	timeWindowsP = CalculateTimeWindows(pair<double, double> {timeWindowPrimary[0], timeWindowPrimary[1]}, pair<double, double> {timeWindowPrimary[2], timeWindowPrimary[3]}, periodP);
	timeWindowsS = CalculateTimeWindows(pair<double, double> {timeWindowSecondary[0], timeWindowSecondary[1]}, pair<double, double> {timeWindowSecondary[2], timeWindowSecondary[3]}, periodS);

	int i = 0;
	for (pair<double, double> window : timeWindowsP)
	{
		if ((window.first <= timeWindowsS[i].first) & (window.second > timeWindowsS[i].second))
		{
			candidateTime = timeWindowsS[i].first + (window.second - timeWindowsS[i].first) / 2;
			timeList.push_back(candidateTime);
			continue;
		}

		// Loop up to matched point in list
		while (window.first > timeWindowsS[i].first)
			{
				i++;
				if (i == timeWindowsS.size())
					break;
			}

		// Check for overlap
		if ((i > 0) & (window.first < timeWindowsS[i - 1].second))
		{
			candidateTime = window.first + (timeWindowsS[i - 1].second - window.first) / 2;
			timeList.push_back(candidateTime);
		}

		else if ((i < timeWindowsS.size()) & (window.second > timeWindowsS[i].first))
		{
			candidateTime = timeWindowsS[i].first + (window.second - timeWindowsS[i].first) / 2;
			timeList.push_back(candidateTime);
		}
	}

	return timeList;
}

vector<double> HootsFilter::CoplanarFilter(CollisionPair objectPair, double timeStep)
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
	return candidateTimeList;
}

vector<double> HootsFilter::DetermineCollisionTimes(CollisionPair objectPair, vector<double> candidateTimeList)
{
	vector<double> collideTimeList;
	double closeTime;
	// TODO - Collision Times
	for (double candidateTime : candidateTimeList)
	{
		closeTime = CalculateClosestApproachTime(objectPair, candidateTime);

		if ((objectPair.GetBoundingRadii() + collisionThreshold) > objectPair.CalculateSeparationAtTime(closeTime))
			collideTimeList.push_back(closeTime);
	}
	return collideTimeList;
}

vector<pair<double, double>>  HootsFilter::CalculateTimeWindows(pair<double, double> window, pair<double, double> window2, double period)
{
	vector<pair<double, double>> windowList;
	// Time windows
	while (window.second < timeStep)
	{
		windowList.push_back(window);
		if (window2.second < timeStep)
			windowList.push_back(window2);
		
		else if (window2.first < timeStep)
		{
			window2.second = timeStep;
			windowList.push_back(window2);
		}

		window.first += period;
		window.second += period;
		window2.first += period;
		window2.second += period;
	}

	if (window.first < timeStep)
	{
		window.second = timeStep;
		windowList.push_back(window);
	}
	return windowList;
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


