#include "stdafx.h"
#include "CollisionPair.cuh"

CollisionPair::CollisionPair()
{
}

CollisionPair::CollisionPair(DebrisObject& objectI, DebrisObject& objectJ)
{
	primaryElements = objectI.GetElements();
	secondaryElements = objectJ.GetElements();
	primaryID = objectI.GetID();
	secondaryID = objectJ.GetID();
	primaryAnomaly = objectI.GetEpochAnomaly();
	secondaryAnomaly = objectJ.GetEpochAnomaly();
	approachAnomalyP = primaryElements.GetTrueAnomaly();
	approachAnomalyS = secondaryElements.GetTrueAnomaly();
	primaryMass = objectI.GetMass();
	secondaryMass = objectJ.GetMass();
	//CalculateRelativeInclination();
	//CalculateArgumenstOfIntersection();
	boundingRadii = (objectI.GetRadius() + objectJ.GetRadius()) * 0.001;
	overlapCount = 1;

	CalculateRelativeInclination();
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

CUDA_CALLABLE_MEMBER void CollisionPair::SetCollisionPair(DebrisObject objectI, DebrisObject objectJ)
{

	primaryElements = objectI.GetElements();
	secondaryElements = objectJ.GetElements();
	primaryID = objectI.GetID();
	secondaryID = objectJ.GetID();
	primaryAnomaly = objectI.GetEpochAnomaly();
	secondaryAnomaly = objectJ.GetEpochAnomaly();
	primaryMass = objectI.GetMass();
	secondaryMass = objectJ.GetMass();
	//CalculateRelativeInclination();
	//CalculateArgumenstOfIntersection();
	boundingRadii = (objectI.GetRadius() + objectJ.GetRadius()) * 0.001;
	overlapCount = 1;

	CalculateRelativeInclination();
}

vector<double> CollisionPair::CalculateAngularWindowPrimary(double distance)
{
	return CalculateAngularWindow(primaryElements, distance, deltaPrimary);
}

vector<double> CollisionPair::CalculateAngularWindowSecondary(double distance)
{
	return CalculateAngularWindow(secondaryElements, distance, deltaSecondary);
}

vector3D CollisionPair::GetPrimaryPositionAtTime(double timeFromEpoch)
{
	// position at time
	double meanAnomaly = TauRange(primaryAnomaly + Tau * timeFromEpoch / primaryElements.CalculatePeriod());
	primaryElements.SetMeanAnomaly(meanAnomaly);
	return primaryElements.GetPosition();
}

vector3D CollisionPair::GetPrimaryVelocityAtTime(double timeFromEpoch)
{
	// velcoity at time
	double meanAnomaly = TauRange(primaryAnomaly + Tau * timeFromEpoch / primaryElements.CalculatePeriod());
	primaryElements.SetMeanAnomaly(meanAnomaly);
	return primaryElements.GetVelocity();
}

vector3D CollisionPair::GetSecondaryPositionAtTime(double timeFromEpoch)
{
	// position at time
	double meanAnomaly = TauRange(secondaryAnomaly + Tau * timeFromEpoch / secondaryElements.CalculatePeriod());
	secondaryElements.SetMeanAnomaly(meanAnomaly);
	return secondaryElements.GetPosition();
}

vector3D CollisionPair::GetSecondaryVelocityAtTime(double timeFromEpoch)
{
	// velcoity at time
	double meanAnomaly = TauRange(secondaryAnomaly + Tau * timeFromEpoch / secondaryElements.CalculatePeriod());
	secondaryElements.SetMeanAnomaly(meanAnomaly);
	return secondaryElements.GetVelocity();
}

void CollisionPair::GenerateArgumenstOfIntersection()
{
	CalculateRelativeInclination();
	coplanar = (relativeInclination <= (2 * asin(boundingRadii / (primaryElements.semiMajorAxis + secondaryElements.semiMajorAxis))));
	if (coplanar)
		CalculateArgumenstOfIntersectionCoplanar();
	else
		CalculateArgumenstOfIntersection();
}

double CollisionPair::CalculateMinimumSeparation_MOID()
{
	moid_data_t mdata;

 	double distance = find_moid_full(primaryElements, secondaryElements, &mdata);

	approachAnomalyP = mdata.obj1_true_anom;
	approachAnomalyS = mdata.obj2_true_anom;
	primaryElements.SetTrueAnomaly(mdata.obj1_true_anom);
	secondaryElements.SetTrueAnomaly(approachAnomalyS);

	SetCollisionAltitude(primaryElements.GetRadialPosition());

	return distance;
}

double CollisionPair::CalculateMinimumSeparation_DL()
{
	COrbitData<double> object1(primaryElements.semiMajorAxis, primaryElements.eccentricity, primaryElements.inclination, primaryElements.rightAscension, primaryElements.argPerigee);
	COrbitData<double> object2(secondaryElements.semiMajorAxis, secondaryElements.eccentricity, secondaryElements.inclination, secondaryElements.rightAscension, secondaryElements.argPerigee);

	double max_root_error, min_root_error, max_anom_error;
	detect_suitable_options(max_root_error, min_root_error, max_anom_error);
	SMOIDResult<double> result = MOID_fast(object1, object2, max_root_error, min_root_error);

	if (!result.good) {
		result = MOID_fast(object2, object1, max_root_error, min_root_error);

		if (!result.good) {
			unsigned int densities[4] = { 1000, 30, 3, 0 };
			result = MOID_direct_search(object1, object2, densities, 0.01, max_anom_error);
		}
	}
	primaryElements.SetEccentricAnomaly(result.u1);
	secondaryElements.SetEccentricAnomaly(result.u2);

	approachAnomalyP = primaryElements.GetTrueAnomaly();
	approachAnomalyS = secondaryElements.GetTrueAnomaly();
	SetCollisionAltitude(primaryElements.GetRadialPosition());

	return result.distance;
}


double CollisionPair::CalculateMinimumSeparation()
{
	double trueAnomalyP, trueAnomalyS, seperation,  eP, eS;

	trueAnomalyP = TauRange(deltaPrimary - primaryElements.argPerigee);
	trueAnomalyS = TauRange(deltaSecondary - secondaryElements.argPerigee);
	eP = primaryElements.eccentricity;
	eS = secondaryElements.eccentricity;
	primaryElements.SetTrueAnomaly(trueAnomalyP);
	secondaryElements.SetTrueAnomaly(trueAnomalyS);

	seperation = primaryElements.GetPosition().CalculateRelativeVector(secondaryElements.GetPosition()).vectorNorm();

	// Find closest approach for elliptical orbits
	if (eP != 0 || eS != 0)
	{
		int error1, error2;
		double seperation1, seperation2, baseSeperation2, altTrueAnomalyS1, altTrueAnomalyP1, altTrueAnomalyS2, altTrueAnomalyP2;
		auto NewtonSeperation = [&](double &trueP, double &trueS)
		{

			int it = 0;
			double F, G, FdfP, FdfS, GdfP, GdfS;
			double uRP, uRS, A, B, C, D, axP, ayP, axS, ayS;
			double rP, rS, sinURP, sinURS, cosURP, cosURS, EP, ES, sinVP, cosVP, sinVS, cosVS;
			double tempAnomalyP, tempAnomalyS, circularAnomalyP, circularAnomalyS, cosRI;
			double k = 2.0, h = 2.0;
			double base, baseMin = 999;

			circularAnomalyP = tempAnomalyP = trueP;
			circularAnomalyS = tempAnomalyS = trueS;
			cosRI = cos(relativeInclination);

			axP = eP * cos(-circularAnomalyP);
			ayP = sqrt(eP * eP - axP * axP); //eP * sin(-circularAnomalyP);
			axS = eS * cos(-circularAnomalyS);
			ayS = sqrt(eS * eS - axS * axS); // eS * sin(-circularAnomalyS);

											 // Min Sep newton method
			while ((abs(h) >= NEWTONTOLERANCE || abs(k) >= NEWTONTOLERANCE) && (it < NEWTONMAXITERATIONS))
			{
				rP = primaryElements.GetRadialPosition(tempAnomalyP);
				rS = secondaryElements.GetRadialPosition(tempAnomalyS);
				uRP = tempAnomalyP - circularAnomalyP;
				uRS = tempAnomalyS - circularAnomalyS;

				sinURP = sin(uRP);
				cosURP = sqrt(1 - sinURP * sinURP);  //cos(uRP);
				sinURS = sin(uRS);
				cosURS = sqrt(1 - sinURS * sinURS);  //cos(uRS);
				sinVP = sin(tempAnomalyP);
				cosVP = sqrt(1 - sinVP * sinVP); // cos(tempAnomalyP);
				sinVS = sin(tempAnomalyS);
				cosVS = sqrt(1 - sinVS * sinVS); // cos(tempAnomalyS);

				A = sinURP + ayP;
				C = sinURS + ayS;
				B = cosURP + axP;
				D = cosURS + axS;

				EP = atan2(sqrt(1 - eP * eP) * sinVP, eP + cosVP);
				ES = atan2(sqrt(1 - eS * eS) * sinVS, eS + cosVS);

				F = rP * eP * sinVP + rS * (A * cosURS - B * cosRI * sinURS);
				G = rS * eS * sinVS + rP * (C * cosURP - D * cosRI * sinURP);

				FdfP = rP * eP * cos(EP) + rS * (cosURP * cosURS + sinURP * sinURS * cosRI);
				FdfS = -rS / (1 + eS * cosVS) * (A * C + B * D * cosRI);
				GdfP = -rP / (1 + eP * cosVP) * (A * C + B * D * cosRI);
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
				++it;
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
		
		altTrueAnomalyP1 = trueAnomalyP;
		altTrueAnomalyS1 = trueAnomalyS;
		error1 = NewtonSeperation(altTrueAnomalyS1, altTrueAnomalyS2);
		primaryElements.SetTrueAnomaly(altTrueAnomalyS1);
		secondaryElements.SetTrueAnomaly(altTrueAnomalyS2);
		
		seperation1 = primaryElements.GetPosition().CalculateRelativeVector(secondaryElements.GetPosition()).vectorNorm();
		if (!error1 && seperation1 < seperation)
		{
			seperation = seperation1;
			trueAnomalyP = altTrueAnomalyP1;
			trueAnomalyS = altTrueAnomalyS2;
		}

		// Test second intersection point

		if (error1 || coplanar)
		{
		altTrueAnomalyP2 = TauRange(deltaPrimary2 - primaryElements.argPerigee);
		altTrueAnomalyS2 = TauRange(deltaSecondary2 - secondaryElements.argPerigee);
		}
		else
		{
		altTrueAnomalyP2 = TauRange(trueAnomalyP + Pi);
		altTrueAnomalyS2 = TauRange(trueAnomalyS + Pi);
		}

		primaryElements.SetTrueAnomaly(altTrueAnomalyP2);
		secondaryElements.SetTrueAnomaly(altTrueAnomalyS2);
		baseSeperation2 = primaryElements.GetPosition().CalculateRelativeVector(secondaryElements.GetPosition()).vectorNorm();
		if (baseSeperation2 < seperation)
		{
			seperation = baseSeperation2;
			trueAnomalyP = altTrueAnomalyP2;
			trueAnomalyS = altTrueAnomalyS2;
		}

		error2 = NewtonSeperation(altTrueAnomalyP2, altTrueAnomalyS2);
		primaryElements.SetTrueAnomaly(altTrueAnomalyP2);
		secondaryElements.SetTrueAnomaly(altTrueAnomalyS2);
		seperation2 = primaryElements.GetPosition().CalculateRelativeVector(secondaryElements.GetPosition()).vectorNorm();

		if (!error2 && seperation2 < seperation)
		{
			seperation = seperation2;
			trueAnomalyP = altTrueAnomalyP2;
			trueAnomalyS = altTrueAnomalyS2;
		}
		else {
			primaryElements.SetTrueAnomaly(trueAnomalyP);
			secondaryElements.SetTrueAnomaly(trueAnomalyS);
		}
	}

	approachAnomalyP = trueAnomalyP;
	approachAnomalyS = trueAnomalyS;
	SetCollisionAltitude(primaryElements.GetRadialPosition());
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

void CollisionPair::SetRelativeVelocity(double relV)
{
	relativeVelocity = relV;
}

double CollisionPair::GetRelativeVelocity()
{
	return relativeVelocity;
}

double CollisionPair::CalculateSeparationAtTime(double timeFromEpoch)
{
	double seperation;
	vector3D positionP = GetPrimaryPositionAtTime(timeFromEpoch);
	vector3D positionS = GetSecondaryPositionAtTime(timeFromEpoch);

	//closest approach distance
	seperation = positionP.CalculateRelativeVector(positionS).vectorNorm();
	collisionAltitude = positionP.vectorNorm();
	return seperation;
}

double CollisionPair::GetMinSeparation()
{
	double seperation;
	primaryElements.SetTrueAnomaly(approachAnomalyP);
	secondaryElements.SetTrueAnomaly(approachAnomalyS);
	vector3D positionP = primaryElements.GetPosition();
	vector3D positionS = secondaryElements.GetPosition();

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
	vector3D hP = primaryElements.GetNormalVector();
	vector3D hS = secondaryElements.GetNormalVector();
	double k = hP.VectorCrossProduct(hS).vectorNorm();

	relativeInclination = asin(k);
}


void CollisionPair::CalculateArgumenstOfIntersection()
{
	// Arguments of intersection
	double cscIr, sinIp, sinIs, cosIp, cosIs, sinOmDif, cosOmDif, XP, XS, YP, YS;

	cscIr = 1 / sin(relativeInclination);
	sinIp = sin(primaryElements.inclination);
	cosIp = cos(primaryElements.inclination);
	sinIs = sin(secondaryElements.inclination);
	cosIs = cos(secondaryElements.inclination);
	sinOmDif = sin(primaryElements.rightAscension - secondaryElements.rightAscension);
	cosOmDif = cos(primaryElements.rightAscension - secondaryElements.rightAscension);

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

	deltaPrimary2 =  TauRange(deltaPrimary + Pi);
	deltaSecondary2 = TauRange(deltaSecondary + Pi);

	approachAnomalyP = TauRange(deltaPrimary - primaryElements.argPerigee);
	approachAnomalyS = TauRange(deltaSecondary - secondaryElements.argPerigee);
}

void CollisionPair::CalculateArgumenstOfIntersectionCoplanar()
{
	double cP, cS, A, B, C, X, X2, Yplus, Yminus;

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
	deltaPrimary = deltaSecondary = TauRange(X);
	deltaPrimary2 = deltaSecondary2 = TauRange(X2);

	approachAnomalyP = TauRange(deltaPrimary - primaryElements.argPerigee);
	approachAnomalyS = TauRange(deltaSecondary - secondaryElements.argPerigee);
}

vector<double> CollisionPair::CalculateAngularWindow(OrbitalElements & elements, double distance, double delta)
{
	vector<double> angleWindows;
	double circularAnomaly, alpha, aX, aY, Q, Qroot, cosUrMinus, cosUrPlus, windowStart, windowEnd, windowStart2, windowEnd2;

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
		cosUrMinus = 0;
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

	angleWindows.insert(angleWindows.end(), { windowStart, windowEnd });
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

		angleWindows.insert(angleWindows.end(), { windowStart2, windowEnd2 });
	}

	return angleWindows;
}
