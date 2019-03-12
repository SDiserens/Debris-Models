
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
	// TODO -  Set objects to position at close approach time and return seperation
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
	boundingRadii = (primary.GetRadius() + secondary.GetRadius()) * 0.001;
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

vector<double> CollisionPair::CalculateAngularWindowPrimary(double distance)
{
	return CalculateAngularWindow(primary, distance, deltaPrimary);
}

vector<double> CollisionPair::CalculateAngularWindowSecondary(double distance)
{
	return CalculateAngularWindow(secondary, distance, deltaSecondary);
}

vector3D CollisionPair::GetPrimaryPositionAtTime(double timeFromEpoch)
{
	// position at time
	double meanAnomaly = fmod(primary.GetEpochAnomaly() + Tau * timeFromEpoch / primary.GetPeriod(), Tau);
	primary.SetMeanAnomaly(meanAnomaly);
	return primary.GetPosition();
}

vector3D CollisionPair::GetPrimaryVelocityAtTime(double timeFromEpoch)
{
	// velcoity at time
	double meanAnomaly = fmod(primary.GetEpochAnomaly() + Tau * timeFromEpoch / primary.GetPeriod(), Tau);
	primary.SetMeanAnomaly(meanAnomaly);
	return primary.GetVelocity();
}

vector3D CollisionPair::GetSecondaryPositionAtTime(double timeFromEpoch)
{
	// position at time
	double meanAnomaly = fmod(secondary.GetEpochAnomaly() + Tau * timeFromEpoch / secondary.GetPeriod(), Tau);
	secondary.SetMeanAnomaly(meanAnomaly);
	return secondary.GetPosition();
}

vector3D CollisionPair::GetSecondaryVelocityAtTime(double timeFromEpoch)
{
	// velcoity at time
	double meanAnomaly = fmod(secondary.GetEpochAnomaly() + Tau * timeFromEpoch / secondary.GetPeriod(), Tau);
	secondary.SetMeanAnomaly(meanAnomaly);
	return secondary.GetVelocity();
}

double CollisionPair::CalculateMinimumSeparation()
{
	double trueAnomalyP, trueAnomalyS, circularAnomalyP, circularAnomalyS, cosRI, seperation, altSeperation, eP, eS;
	OrbitalElements primaryElements(primary.GetElements());
	OrbitalElements secondaryElements(secondary.GetElements());

	trueAnomalyP = deltaPrimary - primaryElements.argPerigee;
	trueAnomalyS = deltaSecondary - secondaryElements.argPerigee;
	eP = primary.GetElements().eccentricity;
	eS = secondary.GetElements().eccentricity;

	// Find closest approach for elliptical orbits
	if (eP != 0 || eS != 0)
	{
		int it = 0;
		double F, G, FdfP, FdfS, GdfP, GdfS;
		double uRP, uRS, A, B, C, D, axP, ayP, axS, ayS;
		double rP, rS, sinURP, sinURS, cosURP, cosURS, EP, ES;
		double tempAnomalyP, tempAnomalyS;
		double k, h = 1.0;

		circularAnomalyP = tempAnomalyP = trueAnomalyP;
		circularAnomalyS = tempAnomalyS = trueAnomalyS;
		cosRI = cos(relativeInclination);

		//Todo - Min Sep newton method
		while ((abs(h) >= NEWTONTOLERANCE || abs(k) >= NEWTONTOLERANCE) && (it < NEWTONMAXITERATIONS))
		{
			rP = primaryElements.GetRadialPosition(tempAnomalyP);
			rS = secondaryElements.GetRadialPosition(tempAnomalyS);
			uRP = tempAnomalyP - circularAnomalyP;
			uRS = tempAnomalyS - circularAnomalyS;

			sinURP = sin(uRP);
			cosURP = cos(uRP);
			sinURS = sin(uRS);
			cosURS = cos(uRS);

			axP = eP * cos(-circularAnomalyP);
			ayP = eP * sin(-circularAnomalyP);
			axS = eS * cos(-circularAnomalyS);
			ayS = eS * sin(-circularAnomalyS);

			A = sinURP + ayP;
			C = sinURS + ayS;
			B = cosURP + axP;
			D = cosURS + axS;

			EP = atan2(sqrt(1 - eP * eP) * sin(tempAnomalyP), eP + cos(tempAnomalyP));
			ES = atan2(sqrt(1 - eS * eS) * sin(tempAnomalyS), eS + cos(tempAnomalyS));

			F = rP * eP * sin(tempAnomalyP) + rS * (A * cosURS - B * cosRI * sinURS);
			G = rS * eS * sin(tempAnomalyS) + rP * (C * cosURP - D * cosRI * sinURP);

			FdfP = rP * eP * cos(EP) + rS * (cosURP * cosURS + sinURP * sinURS * cosRI);
			FdfS = -rS / (1 + eS * cos(tempAnomalyS)) * (A * C + B * D * cosRI);
			GdfP = -rP / (1 + eP * cos(tempAnomalyP)) * (A * C + B * D * cosRI);
			GdfS = rS * eS * cos(ES) + rP * (cosURP * cosURS + sinURP * sinURS * cosRI);


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

	primaryElements.SetTrueAnomaly(trueAnomalyP);
	secondaryElements.SetTrueAnomaly(trueAnomalyS);
	seperation = primaryElements.GetPosition().CalculateRelativeVector(secondaryElements.GetPosition()).vectorNorm();


	primaryElements.SetTrueAnomaly(fmod(trueAnomalyP + Pi, Tau));
	secondaryElements.SetTrueAnomaly(fmod(trueAnomalyS + Pi, Tau));
	altSeperation = primaryElements.GetPosition().CalculateRelativeVector(secondaryElements.GetPosition()).vectorNorm();

	if (altSeperation < seperation)
	{
		seperation = altSeperation;
		trueAnomalyP = fmod(trueAnomalyP + Pi, Tau);
		trueAnomalyS = fmod(trueAnomalyS + Pi, Tau);
	}
	approachAnomalyP = trueAnomalyP;
	approachAnomalyS = trueAnomalyS;

	return seperation;
}

double CollisionPair::GetBoundingRadii()
{
	return boundingRadii; // Combined radii in kilometres;
}

double CollisionPair::CalculateSeparationAtTime(double timeFromEpoch)
{
	double seperation;
	vector3D positionP = GetPrimaryPositionAtTime(timeFromEpoch);
	vector3D positionS = GetSecondaryPositionAtTime(timeFromEpoch);

	//TODO - closest approach distance
	seperation =positionP.CalculateRelativeVector(positionS).vectorNorm();
	return seperation;
}

void CollisionPair::CalculateRelativeInclination()
{
	//TODO - Calculate relative inclination
	/*
	sin IR = |cross(hP, hC)|

	where r hP is the normal to the orbit plane of the primary object
	*/
	vector3D hP = primary.GetNormalVector();
	vector3D hS = secondary.GetNormalVector();
	double k = hP.VectorDotProduct(hS);

	relativeInclination = acos(k);
}


void CollisionPair::CalculateArgumenstOfIntersection()
{
	// Arguments of intersection
	double cscIr, sinIp, sinIs, sinOmDif;

	cscIr = 1 / sin(relativeInclination);
	sinIp = sin(primary.GetElements().inclination);
	sinIs = sin(secondary.GetElements().inclination);
	sinOmDif = sin(primary.GetElements().rightAscension - secondary.GetElements().rightAscension);

	deltaPrimary = asin(cscIr * sinIs * sinOmDif);
	deltaSecondary = asin(cscIr * sinIp * sinOmDif);

}

void CollisionPair::CalculateArgumenstOfIntersectionCoplanar()
{
	// TODO Coplanar Arguments of intersection
	deltaPrimary = 0;
	deltaSecondary = 0;
}

vector<double> CollisionPair::CalculateAngularWindow(DebrisObject & object, double distance, double delta)
{
	vector<double> timeWindows;
	double circularAnomaly, alpha, aX, aY, Q, Qroot, cosUrMinus, cosUrPlus, windowStart, windowEnd, windowStart2, windowEnd2;

	OrbitalElements elements(object.GetElements());
	//TODO - Calculate Angular Windows
	circularAnomaly = delta - elements.argPerigee;
	alpha = elements.semiMajorAxis * (1 - elements.eccentricity * elements.eccentricity) * sin(relativeInclination);
	aX = elements.eccentricity * cos(-circularAnomaly);
	aY = elements.eccentricity * sin(-circularAnomaly);
	Q = alpha * (alpha - 2 * distance * aY) - (1 - elements.eccentricity * elements.eccentricity) * distance * distance;

	if (Q >= 0)
		Qroot = sqrt(Q);
	else
	{
		timeWindows.push_back(-1.0);
		return timeWindows;
	}

	cosUrMinus = (-distance * distance * aX - (alpha - distance * aY) * Qroot) / (Q + distance * distance);
	cosUrPlus = (-distance * distance * aX + (alpha - distance * aY) * Qroot) / (Q + distance * distance);

	if (abs(cosUrMinus) > 1)
	{
		timeWindows.push_back(-2.0);
		return timeWindows;
	}

	else if (abs(cosUrPlus) > 1)
	{
		timeWindows.push_back(-3.0);
		return timeWindows;
	}


	windowStart = acos(cosUrMinus);
	windowEnd = acos(cosUrPlus);

	windowStart2 = Tau - windowEnd;
	windowEnd2 = Tau - windowStart;

	if (windowEnd < windowStart)
	{
		double temp = windowEnd;
		windowEnd = windowEnd2;
		windowEnd2 = temp;
	}
	//TODO - check for singular case
	timeWindows.insert(timeWindows.end(), { windowStart, windowEnd, windowStart2, windowEnd2 });

	return timeWindows;
}
