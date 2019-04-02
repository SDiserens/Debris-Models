
#include "stdafx.h"
#include "Collisions.h"



double CollisionAlgorithm::GetElapsedTime()
{
	return elapsedTime;
}

vector<CollisionPair> CollisionAlgorithm::CreatePairList(DebrisPopulation & population)
{
	vector<CollisionPair> pairList;
	// For each object in population -
	for (auto it=population.population.begin(); it!= population.population.end(); it++)
	{
		// For each subsequent object
		auto jt = it;
		for (jt++; jt != population.population.end(); ++jt)
		{
			/// Add pair to list
			//DebrisObject& primaryObject(population.Ge), secondaryObject;
			pairList.push_back(CollisionPair(it->second, jt->second));
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

/*
double CollisionAlgorithm::CalculateClosestApproach(CollisionPair objectPair)
{
	//  Set objects to position at close approach time and return seperation
	return objectPair.CalculateSeparationAtTime();
}
*/

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
	//CalculateRelativeInclination();
	//CalculateArgumenstOfIntersection();
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
	double meanAnomaly = TauRange(primary.GetEpochAnomaly() + Tau * timeFromEpoch / primary.GetPeriod());
	primary.SetMeanAnomaly(meanAnomaly);
	return primary.GetPosition();
}

vector3D CollisionPair::GetPrimaryVelocityAtTime(double timeFromEpoch)
{
	// velcoity at time
	double meanAnomaly = TauRange(primary.GetEpochAnomaly() + Tau * timeFromEpoch / primary.GetPeriod());
	primary.SetMeanAnomaly(meanAnomaly);
	return primary.GetVelocity();
}

vector3D CollisionPair::GetSecondaryPositionAtTime(double timeFromEpoch)
{
	// position at time
	double meanAnomaly = TauRange(secondary.GetEpochAnomaly() + Tau * timeFromEpoch / secondary.GetPeriod());
	secondary.SetMeanAnomaly(meanAnomaly);
	return secondary.GetPosition();
}

vector3D CollisionPair::GetSecondaryVelocityAtTime(double timeFromEpoch)
{
	// velcoity at time
	double meanAnomaly = TauRange(secondary.GetEpochAnomaly() + Tau * timeFromEpoch / secondary.GetPeriod());
	secondary.SetMeanAnomaly(meanAnomaly);
	return secondary.GetVelocity();
}

double CollisionPair::CalculateMinimumSeparation()
{
	double trueAnomalyP, trueAnomalyS, seperation, altSeperation, eP, eS;
	OrbitalElements primaryElements(primary.GetElements());
	OrbitalElements secondaryElements(secondary.GetElements());

	trueAnomalyP = TauRange(deltaPrimary - primaryElements.argPerigee);
	trueAnomalyS = TauRange(deltaSecondary - secondaryElements.argPerigee);
	eP = primaryElements.eccentricity;
	eS = secondaryElements.eccentricity;
	auto NewtonSeperation = [](double &trueP, double &trueS) {};

	// Find closest approach for elliptical orbits
	if (eP != 0 || eS != 0)
	{
		auto NewtonSeperation = [&](double &trueP, double &trueS)
		{

			int it = 0;
			double F, G, FdfP, FdfS, GdfP, GdfS;
			double uRP, uRS, A, B, C, D, axP, ayP, axS, ayS;
			double rP, rS, sinURP, sinURS, cosURP, cosURS, EP, ES;
			double tempAnomalyP, tempAnomalyS, circularAnomalyP, circularAnomalyS, cosRI;
			double k, h = 1.0;

			circularAnomalyP = tempAnomalyP = trueP;
			circularAnomalyS = tempAnomalyS = trueS;
			cosRI = cos(relativeInclination);

			// Min Sep newton method
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
			trueP = TauRange(tempAnomalyP);
			trueS = TauRange(tempAnomalyS);
		};

		NewtonSeperation(trueAnomalyP, trueAnomalyS);
	}

	primaryElements.SetTrueAnomaly(trueAnomalyP);
	secondaryElements.SetTrueAnomaly(trueAnomalyS);
	seperation = primaryElements.GetPosition().CalculateRelativeVector(secondaryElements.GetPosition()).vectorNorm();


	primaryElements.SetTrueAnomaly(TauRange(trueAnomalyP + Pi));
	secondaryElements.SetTrueAnomaly(TauRange(trueAnomalyS + Pi));

	if (eP != 0 || eS != 0)
	{
		//TODO - Check this is working
		NewtonSeperation(trueAnomalyP, trueAnomalyS);
	}
	altSeperation = primaryElements.GetPosition().CalculateRelativeVector(secondaryElements.GetPosition()).vectorNorm();

	if (altSeperation < seperation)
	{
		seperation = altSeperation;
		trueAnomalyP = TauRange(trueAnomalyP + Pi);
		trueAnomalyS = TauRange(trueAnomalyS + Pi);
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

	//closest approach distance
	seperation = positionP.CalculateRelativeVector(positionS).vectorNorm();
	return seperation;
}

void CollisionPair::CalculateRelativeInclination()
{
	// Calculate relative inclination
	/*
	sin IR = |cross(hP, hC)|

	where r hP is the normal to the orbit plane of the primary object
	*/
	vector3D hP = primary.GetNormalVector();
	vector3D hS = secondary.GetNormalVector();
	double k = hP.VectorCrossProduct(hS).vectorNorm();

	relativeInclination = asin(k);
}


void CollisionPair::CalculateArgumenstOfIntersection()
{
	// Arguments of intersection
	double cscIr, sinIp, sinIs, cosIp, cosIs, sinOmDif, cosOmDif, XP, XS, YP, YS;

	cscIr = 1 / sin(relativeInclination);
	sinIp = sin(primary.GetElements().inclination);
	cosIp = cos(primary.GetElements().inclination);
	sinIs = sin(secondary.GetElements().inclination);
	cosIs = cos(secondary.GetElements().inclination);
	sinOmDif = sin(primary.GetElements().rightAscension - secondary.GetElements().rightAscension);
	cosOmDif = cos(primary.GetElements().rightAscension - secondary.GetElements().rightAscension);

	XP = cscIr*(sinIp*cosIs - sinIs*cosIp*cosOmDif);
	XS = cscIr*(sinIp*cosIs*cosOmDif - sinIs*cosIp);
	YP = cscIr * sinIs * sinOmDif;
	YS = cscIr * sinIp * sinOmDif;

	deltaPrimary = asin(YP);
	if (XP < 0)
		deltaPrimary = Pi - deltaPrimary;
	else if (YP < 0)
		deltaPrimary += Tau;

	deltaSecondary = asin(YS);
	if (XS < 0)
		deltaSecondary = Pi - deltaSecondary;
	else if (YS < 0)
		deltaSecondary += Tau;

}

void CollisionPair::CalculateArgumenstOfIntersectionCoplanar()
{
	// TODO Coplanar Arguments of intersection
	deltaPrimary = 0;
	deltaSecondary = 0;
}

vector<double> CollisionPair::CalculateAngularWindow(DebrisObject & object, double distance, double delta)
{
	vector<double> anlgeWindows;
	double circularAnomaly, alpha, aX, aY, Q, Qroot, cosUrMinus, cosUrPlus, windowStart, windowEnd, windowStart2, windowEnd2;

	OrbitalElements elements(object.GetElements());
	// Calculate Angular Windows
	circularAnomaly = delta - elements.argPerigee;
	alpha = elements.semiMajorAxis * (1 - elements.eccentricity * elements.eccentricity) * sin(relativeInclination);
	aX = elements.eccentricity * cos(-circularAnomaly);
	aY = elements.eccentricity * sin(-circularAnomaly);
	Q = alpha * (alpha - 2 * distance * aY) - (1 - elements.eccentricity * elements.eccentricity) * distance * distance;

	if (Q >= 0)
		Qroot = sqrt(Q);
	else
	{
		anlgeWindows.push_back(-1.0);
		return anlgeWindows;
	}

	cosUrMinus = (-distance * distance * aX - (alpha - distance * aY) * Qroot) / (Q + distance * distance);
	cosUrPlus = (-distance * distance * aX + (alpha - distance * aY) * Qroot) / (Q + distance * distance);

	if (abs(cosUrMinus) > 1)
	{
		anlgeWindows.push_back(-2.0);
		return anlgeWindows;
	}

	else if (abs(cosUrPlus) > 1)
	{
		anlgeWindows.push_back(-3.0);
		return anlgeWindows;
	}


	windowStart = acos(cosUrMinus);
	windowEnd = acos(cosUrPlus);
	if (windowEnd < windowStart)
	{
		// check for singular case where close approach at perigee
		windowStart -= Tau;
		anlgeWindows.insert(anlgeWindows.end(), { windowStart, windowEnd });
	}
	else
	{
		windowStart2 = Tau - windowEnd;
		windowEnd2 = Tau - windowStart;

		anlgeWindows.insert(anlgeWindows.end(), { windowStart, windowEnd, windowStart2, windowEnd2 });
	}
	return anlgeWindows;
}
