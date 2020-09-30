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
	constellation = objectI.GetConstellationID() + objectJ.GetConstellationID();
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

double CollisionPair::CalculateMinimumSeparation_DL(double max_root_error, double min_root_error, double max_anom_error)
{
	COrbitData<double> object1(primaryElements.semiMajorAxis, primaryElements.eccentricity, primaryElements.inclination, primaryElements.argPerigee, primaryElements.rightAscension);
	COrbitData<double> object2(secondaryElements.semiMajorAxis, secondaryElements.eccentricity, secondaryElements.inclination, secondaryElements.argPerigee, secondaryElements.rightAscension);


	SMOIDResult<double> result = MOID_fast(object1, object2, max_root_error, min_root_error);

	if (!result.good) {
		result = MOID_fast(object2, object1, max_root_error, min_root_error);

		if (!result.good) {
			unsigned int densities[4] = { 1000, 30, 3, 0 };
			result = MOID_direct_search(object1, object2, densities, 0.01, max_anom_error);
		}
	}

	primaryElements.SetEccentricAnomaly(TauRange(result.u1_2));
	secondaryElements.SetEccentricAnomaly(TauRange(result.u2_2));
	approachAnomalyP2 = primaryElements.GetTrueAnomaly();
	approachAnomalyS2 = secondaryElements.GetTrueAnomaly();

	double seperation1 = primaryElements.GetPosition().CalculateRelativeVector(secondaryElements.GetPosition()).vectorNorm();
	minSeperation2 = result.distance2;

	primaryElements.SetEccentricAnomaly(TauRange(result.u1));
	secondaryElements.SetEccentricAnomaly(TauRange(result.u2));

	approachAnomalyP = primaryElements.GetTrueAnomaly();
	approachAnomalyS = secondaryElements.GetTrueAnomaly();
	SetCollisionAltitude(primaryElements.GetRadialPosition());
	minSeperation = result.distance;

	return result.distance;
}

double CollisionPair::CalculateLowerBoundSeparation()
{
	double tau, C2, sigma, eP, eS, sinRI, cosRI, A, sigma1, sigma2, sigma3, sigma4;

	eP = primaryElements.eccentricity;
	eS = secondaryElements.eccentricity;
	sinRI = sin(relativeInclination);
	cosRI = cos(relativeInclination);
	A = (1 - eP)*(1 - eS) * sinRI* sinRI;
	primaryElements.SetTrueAnomaly(approachAnomalyP);
	secondaryElements.SetTrueAnomaly(approachAnomalyS);
	sigma1 = (primaryElements.GetPosition() - secondaryElements.GetPosition()).vectorNorm();

	secondaryElements.SetTrueAnomaly(approachAnomalyS2);
	sigma2 = (primaryElements.GetPosition() - secondaryElements.GetPosition()).vectorNorm();

	primaryElements.SetTrueAnomaly(approachAnomalyP2);
	sigma3 = (primaryElements.GetPosition() - secondaryElements.GetPosition()).vectorNorm();

	secondaryElements.SetTrueAnomaly(approachAnomalyS);
	sigma4 = (primaryElements.GetPosition() - secondaryElements.GetPosition()).vectorNorm();

	C2 = A / (A + 2*(1 + abs(cosRI) * sqrt((1 - eP*eP)*(1-eS*eS)) - eP*eS) );
	sigma = min(min(sigma1, sigma2), min(sigma3, sigma4));

	tau = sqrt(C2) * sigma;
	return tau;
}

double CollisionPair::CalculateMinimumSeparation()
{
	double trueAnomalyP, trueAnomalyS, seperation, seperation2,  eP, eS;

	trueAnomalyP = TauRange(deltaPrimary - primaryElements.argPerigee);
	trueAnomalyS = TauRange(deltaSecondary - secondaryElements.argPerigee);
	eP = primaryElements.eccentricity;
	eS = secondaryElements.eccentricity;
	primaryElements.SetTrueAnomaly(trueAnomalyP);
	secondaryElements.SetTrueAnomaly(trueAnomalyS);

	seperation = primaryElements.GetPosition().CalculateRelativeVector(secondaryElements.GetPosition()).vectorNorm();
	
	approachAnomalyP = trueAnomalyP;
	approachAnomalyS = trueAnomalyS;

	// Find closest approach for elliptical orbits
	if (eP != 0 || eS != 0)
	{
		int error1, error2;
		double seperation1, altSeperation2, altTrueAnomalyS1, altTrueAnomalyP1, altTrueAnomalyS2, altTrueAnomalyP2, circularAnomalyP, circularAnomalyS;

		circularAnomalyP = deltaPrimary - primaryElements.argPerigee;
		circularAnomalyS = deltaSecondary - secondaryElements.argPerigee;
		auto NewtonSeperation = [&](double &trueP, double &trueS)
		{
			int it = 0;
			double F, G, FdfP, FdfS, GdfP, GdfS, tolerance, Fold = 1, Gold = 1;
			double uRP, uRS, A, B, C, D, axP, ayP, axS, ayS;
			double rP, rS, sinURP, sinURS, cosURP, cosURS, EP, ES, sinVP, cosVP, sinVS, cosVS;
			double tempAnomalyP, tempAnomalyS, cosRI;
			double k = 2.0, h = 2.0;
			double base;

			tempAnomalyP = trueP;
			tempAnomalyS = trueS;
			cosRI = cos(relativeInclination);

			axP = eP * cos(-circularAnomalyP);
			ayP = sqrt(eP * eP - axP * axP); //eP * sin(-circularAnomalyP);
			axS = eS * cos(-circularAnomalyS);
			ayS = sqrt(eS * eS - axS * axS); // eS * sin(-circularAnomalyS);
			tolerance = NEWTONTOLERANCE;
											 // Min Sep newton method
			while ((abs(h) >= tolerance || abs(k) >= tolerance) && (it < NEWTONMAXITERATIONS))
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
				//if (abs(base) < baseMin)
					//baseMin = abs(base);
				h = (F * GdfS - G * FdfS) / base;
				k = (G * FdfP - F * GdfP) / base;

				//if (it != 0){
				//	if (F * Fold < 0 && h * hOld > 0 && G * Gold < 0 && k * kOld > 0) {
				//		h = -hOld / 2;
				//		k = -kOld / 2;
				//	}
				//}
				//else {
				//	Fold = F;
				//	Gold = G;
				//	hOld = h;
				//	kOld = k;
				//}
				//if (it > 1 && abs(base) < 25)
				//{
					//Implement line search
				//	it = NEWTONMAXITERATIONS;
				//	break;
			//	}

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
				if (abs(F) < 10 && abs(G) < 10)
					return 1;
				else
					return 2;
			else
				return 0;
		};
		
		altTrueAnomalyP1 = trueAnomalyP;
		altTrueAnomalyS1 = trueAnomalyS;
		error1 = NewtonSeperation(altTrueAnomalyP1, altTrueAnomalyS1);
		//if (error1 == 2) {
		//	altTrueAnomalyP1 = trueAnomalyP - step;
		//	altTrueAnomalyS1 = trueAnomalyS - step;
		//	error1 = NewtonSeperation(altTrueAnomalyP1, altTrueAnomalyS1);
		//}
		
		while (error1 == 1)
			error1 = NewtonSeperation(altTrueAnomalyP1, altTrueAnomalyS1);

		primaryElements.SetTrueAnomaly(altTrueAnomalyP1);
		secondaryElements.SetTrueAnomaly(altTrueAnomalyS1);
		
		seperation1 = primaryElements.GetPosition().CalculateRelativeVector(secondaryElements.GetPosition()).vectorNorm();
		if (seperation1 < seperation)
		{
			seperation = seperation1;
			trueAnomalyP = altTrueAnomalyP1;
			trueAnomalyS = altTrueAnomalyS1;
		}

		minSeperation = seperation;
		approachAnomalyP = trueAnomalyP;
		approachAnomalyS = trueAnomalyS;

		// Test second intersection point

		if (error1==2 || coplanar)
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
		seperation2 = primaryElements.GetPosition().CalculateRelativeVector(secondaryElements.GetPosition()).vectorNorm();

		altTrueAnomalyP2 = trueAnomalyP;
		altTrueAnomalyS2 = trueAnomalyS;
		error2 = NewtonSeperation(altTrueAnomalyP2, altTrueAnomalyS2);
		//if (error2 == 2) {
		//	altTrueAnomalyP2 = trueAnomalyP - step;
		//	altTrueAnomalyS2 = trueAnomalyS - step;
		//	error2 = NewtonSeperation(altTrueAnomalyP2, altTrueAnomalyS2);
		//}
		

		primaryElements.SetTrueAnomaly(altTrueAnomalyP2);
		secondaryElements.SetTrueAnomaly(altTrueAnomalyS2);
		altSeperation2 = primaryElements.GetPosition().CalculateRelativeVector(secondaryElements.GetPosition()).vectorNorm();

		if (altSeperation2 < seperation2 && abs(altTrueAnomalyP2 - altTrueAnomalyP1) > 0.001)
		{
			seperation2 = altSeperation2;
			trueAnomalyP = altTrueAnomalyP2;
			trueAnomalyS = altTrueAnomalyS2;
		}
		else {
			primaryElements.SetTrueAnomaly(trueAnomalyP);
			secondaryElements.SetTrueAnomaly(trueAnomalyS);
		}
	}
	else {
		seperation2 = seperation;
		trueAnomalyP = TauRange(trueAnomalyP + Pi);
		trueAnomalyS = TauRange(trueAnomalyS + Pi);
	}


	minSeperation2 = seperation2;
	approachAnomalyP2 = trueAnomalyP;
	approachAnomalyS2 = trueAnomalyS;

	return min(seperation, seperation2);
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

	deltaPrimary = TauRange(atan2(YP, XP));
	deltaSecondary = TauRange(atan2(YS, XS));
	//deltaPrimary = asin(YP);
	//if (XP < 0)
	//	deltaPrimary = Pi - deltaPrimary;
	//else if (YP < 0)
	//	deltaPrimary += Tau;

	//deltaSecondary = asin(YS);
	//if (XS < 0)
	//	deltaSecondary = Pi - deltaSecondary;
	//else if (YS < 0)
	//	deltaSecondary += Tau;

	deltaPrimary2 =  TauRange(deltaPrimary + Pi);
	deltaSecondary2 = TauRange(deltaSecondary + Pi);

	approachAnomalyP = TauRange(deltaPrimary - primaryElements.argPerigee);
	approachAnomalyS = TauRange(deltaSecondary - secondaryElements.argPerigee);
	approachAnomalyP2 = TauRange(deltaPrimary2 - primaryElements.argPerigee);
	approachAnomalyS2 = TauRange(deltaSecondary2 - secondaryElements.argPerigee);
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
