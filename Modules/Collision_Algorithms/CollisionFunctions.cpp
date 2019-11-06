
#include "stdafx.h"
#include "Collisions.h"



double CollisionAlgorithm::GetElapsedTime()
{
	return elapsedTime;
}

vector<double> CollisionAlgorithm::GetCollisionVerbose()
{
	return GetCollisionProbabilities();
}

vector<double> CollisionAlgorithm::GetNewCollisionVerbose()
{
	return GetNewCollisionProbabilities();
}

vector<double> CollisionAlgorithm::GetNewCollisionAltitudes()
{
	vector<double> newList(newCollisionAltitudes);
	newCollisionAltitudes.clear();
	return newList;
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
			CollisionPair pair(it->second, jt->second);
			if (PerigeeApogeeTest(pair))
				pairList.push_back(pair);
		}
	}

	return pairList;
}

bool CollisionAlgorithm::PerigeeApogeeTest(CollisionPair& objectPair)
{
	double maxPerigee, minApogee;
	// Perigee Apogee Test
	maxPerigee = max(objectPair.primary.GetPerigee(), objectPair.secondary.GetPerigee());
	minApogee = min(objectPair.primary.GetApogee(), objectPair.secondary.GetApogee());

	return (maxPerigee - minApogee) <= pAThreshold;
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
	OrbitalElements primaryElements = primary.GetElements();
	OrbitalElements secondaryElements = secondary.GetElements();

	trueAnomalyP = TauRange(deltaPrimary - primaryElements.argPerigee);
	trueAnomalyS = TauRange(deltaSecondary - secondaryElements.argPerigee);
	eP = primaryElements.eccentricity;
	eS = secondaryElements.eccentricity;
	auto NewtonSeperation = [](double &trueP, double &trueS) {};

	// Find closest approach for elliptical orbits
	if (eP != 0 || eS != 0)
	{
		int error;

		auto NewtonSeperation = [&](double &trueP, double &trueS)
		{

			int it = 0;
			double F, G, FdfP, FdfS, GdfP, GdfS;
			double uRP, uRS, A, B, C, D, axP, ayP, axS, ayS;
			double rP, rS, sinURP, sinURS, cosURP, cosURS, EP, ES;
			double tempAnomalyP, tempAnomalyS, circularAnomalyP, circularAnomalyS, cosRI;
			double k = 2.0, h = 2.0;
			double base, baseMin = 999;

			circularAnomalyP = tempAnomalyP = trueP;
			circularAnomalyS = tempAnomalyS = trueS;
			cosRI = cos(relativeInclination);

			axP = eP * cos(-circularAnomalyP);
			ayP = eP * sin(-circularAnomalyP);
			axS = eS * cos(-circularAnomalyS);
			ayS = eS * sin(-circularAnomalyS);

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

				base = (FdfS*GdfP - FdfP*GdfS);
				if (abs(base) < baseMin)
					baseMin = abs(base);

				h = (F * GdfS - G * FdfS) / base;
				k = (G * FdfP - F * GdfP) / base;


				if (it > 1 && abs(base) < 25)
				{
					//Implement line search
					it = NEWTONMAXITERATIONS;
					break;
				}

				// Update values
				tempAnomalyP = TauRange(tempAnomalyP + h);
				tempAnomalyS = TauRange(tempAnomalyS + k);
				it++;
			}
			//TODO Handle case where iterations reached
			if (it == NEWTONMAXITERATIONS + 1)
				it = it;
			else
			{
				trueP = tempAnomalyP;
				trueS = tempAnomalyS;
			}
			if (it == NEWTONMAXITERATIONS)
				return 1;
			else
				return 0;
		};

		error = NewtonSeperation(trueAnomalyP, trueAnomalyS);

		primaryElements.SetTrueAnomaly(trueAnomalyP);
		secondaryElements.SetTrueAnomaly(trueAnomalyS);
		seperation = primaryElements.GetPosition().CalculateRelativeVector(secondaryElements.GetPosition()).vectorNorm();

		if (error)
		{
			double altTrueAnomalyP, altTrueAnomalyS;
			if (coplanar)
			{
				altTrueAnomalyP = TauRange(deltaPrimary2 - primaryElements.argPerigee);
				altTrueAnomalyS = TauRange(deltaSecondary2 - secondaryElements.argPerigee);
			}
			else
			{
				altTrueAnomalyP = TauRange(trueAnomalyP + Pi);
				altTrueAnomalyS = TauRange(trueAnomalyS + Pi);
			}
			error = NewtonSeperation(trueAnomalyP, trueAnomalyS);

			primaryElements.SetTrueAnomaly(altTrueAnomalyP);
			secondaryElements.SetTrueAnomaly(altTrueAnomalyS);
			altSeperation = primaryElements.GetPosition().CalculateRelativeVector(secondaryElements.GetPosition()).vectorNorm();

			if (error && altSeperation < seperation)
			{
				//throw NewtonConvergenceException();
				trueAnomalyP = altTrueAnomalyP;
				trueAnomalyS = altTrueAnomalyS;
				seperation = altSeperation;
			}
		}

	}

	SetCollisionAltitude(primaryElements.GetRadialPosition());

	// Test second intersection point
	if (coplanar)
	{
		trueAnomalyP = TauRange(deltaPrimary2 - primaryElements.argPerigee);
		trueAnomalyS = TauRange(deltaSecondary2 - secondaryElements.argPerigee);
	}
	else
	{
		trueAnomalyP = TauRange(trueAnomalyP + Pi);
		trueAnomalyS = TauRange(trueAnomalyS + Pi);
	}

	primaryElements.SetTrueAnomaly(trueAnomalyP);
	secondaryElements.SetTrueAnomaly(trueAnomalyS);

	if (eP != 0 || eS != 0)
	{
		NewtonSeperation(trueAnomalyP, trueAnomalyS);
	}
	altSeperation = primaryElements.GetPosition().CalculateRelativeVector(secondaryElements.GetPosition()).vectorNorm();

	if (altSeperation < seperation)
	{
		seperation = altSeperation;
		trueAnomalyP = TauRange(trueAnomalyP + Pi);
		trueAnomalyS = TauRange(trueAnomalyS + Pi);
		SetCollisionAltitude(primaryElements.GetRadialPosition());
	}
	approachAnomalyP = trueAnomalyP;
	approachAnomalyS = trueAnomalyS;

	return seperation;
}

double CollisionPair::GetBoundingRadii()
{
	return boundingRadii; // Combined radii in kilometres;
}

double CollisionPair::GetCollisionAltitude()
{
	return collisionAltitude;
}

void CollisionPair::SetCollisionAltitude(double altitude)
{
	collisionAltitude = altitude;
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

	deltaPrimary2 = deltaPrimary + Pi;
	deltaSecondary2 = deltaSecondary + Pi;
}

void CollisionPair::CalculateArgumenstOfIntersectionCoplanar()
{
	double cP, cS, A, B, C, X, X2, Yplus, Yminus;
	OrbitalElements &primaryElements = primary.GetElements();
	OrbitalElements &secondaryElements = secondary.GetElements();

	cP = primaryElements.semiMajorAxis * (1 - primaryElements.eccentricity * primaryElements.eccentricity);
	cS = secondaryElements.semiMajorAxis * (1 - secondaryElements.eccentricity * secondaryElements.eccentricity);

	A = cP - cS;
	B = cP * secondaryElements.eccentricity * cos(secondaryElements.argPerigee) - cS * primaryElements.eccentricity * cos(primaryElements.argPerigee);
	C = cP * secondaryElements.eccentricity * sin(secondaryElements.argPerigee) - cS * primaryElements.eccentricity * sin(primaryElements.argPerigee);

	Yplus = (C + sqrt(C*C + B*B - A*A)) / (A - B);

	Yminus = (C - sqrt(C*C + B*B - A*A)) / (A - B);

	X = 2 * atan(Yplus);
	X2 = 2 * atan(Yminus);

	// (Rate of change of seperations?)
	deltaPrimary = deltaSecondary = X;
	deltaPrimary2 = deltaSecondary2 = X2;
}

vector<double> CollisionPair::CalculateAngularWindow(DebrisObject & object, double distance, double delta)
{
	vector<double> angleWindows;
	double circularAnomaly, alpha, aX, aY, Q, Qroot, cosUrMinus, cosUrPlus, windowStart, windowEnd, windowStart2, windowEnd2;

	OrbitalElements& elements(object.GetElements());
	// Calculate Angular Windows
	circularAnomaly = delta - elements.argPerigee;
	alpha = elements.semiMajorAxis * (1 - elements.eccentricity * elements.eccentricity) * sin(relativeInclination);
	aX = elements.eccentricity * cos(-circularAnomaly);
	aY = elements.eccentricity * sin(-circularAnomaly);
	Q = alpha * (alpha - 2 * distance * aY) - (1 - elements.eccentricity * elements.eccentricity) * distance * distance;


	if (Q < 0)
	{
		// Handle  coplanar case
		angleWindows.push_back(-1.0);
		return angleWindows;
	}
	else if (Q == 0)
	{
		// Check for singular case where close approach at perige
		Qroot = 0;
		cosUrPlus = -aX;
	}
	else
	{
		Qroot = sqrt(Q);
		cosUrMinus = (-distance * distance * aX - (alpha - distance * aY) * Qroot) / (Q + distance * distance);
		cosUrPlus = (-distance * distance * aX + (alpha - distance * aY) * Qroot) / (Q + distance * distance);
	}

	// Handle  coplanar case
	if (abs(cosUrMinus) > 1)
	{
		angleWindows.push_back(-2.0);
		return angleWindows;
	}
	else if (abs(cosUrPlus) > 1)
	{
		angleWindows.push_back(-3.0);
		return angleWindows;
	}

	windowEnd = acos(cosUrPlus);
	windowStart = 0 - windowEnd;

	windowStart -= circularAnomaly;
	windowEnd -= circularAnomaly;

	if (windowEnd < 0)
	{
		windowStart += Tau;
		windowEnd += Tau;
	}

	angleWindows.insert(angleWindows.end(), {windowStart, windowEnd });
	if (Q != 0)
		{
			windowStart2 = acos(cosUrMinus);
			windowEnd2 = Tau - windowStart2;
			windowStart2 -= circularAnomaly;
			windowEnd2 -= circularAnomaly;

			if (windowEnd2 < 0)
			{
				windowStart2 += Tau;
				windowEnd2 += Tau;
			}

			angleWindows.insert(angleWindows.end(), {windowStart2, windowEnd2 });
		}

	return angleWindows;
}
