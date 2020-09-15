#include "stdafx.h"

int DebrisObject::objectSEQ = 0;

DebrisObject::DebrisObject()
{
	positionSync = velocitySync = false;
	periodSync = false;
	noradID = -1;
}

DebrisObject::DebrisObject(double init_radius, double init_mass, double init_length, double semiMajorAxis, double eccentricity, double inclination,
	double rightAscension, double argPerigee, double init_meanAnomaly, int type)
{
	objectID = ++objectSEQ; // This should be generating a unique ID per object incl. when threading (needs revision for multi-thread)
	sourceID = objectID;
	parentID = 0;
	radius = init_radius;
	area = Pi * radius * radius;
	mass = init_mass;
	CalculateAreaToMass();
	length = init_length;
	elements = OrbitalElements(semiMajorAxis, eccentricity, inclination, TauRange(rightAscension), TauRange(argPerigee), TauRange(init_meanAnomaly));
	meanAnomalyEpoch = init_meanAnomaly;
	nFrag = 1;
	objectType = type;
	positionSync = velocitySync = false;
	periodSync = false;
	coefficientDrag = 2.2;
	bStar = NAN;

	noradID = -1;
	switch (type) {
	case 0:
		isIntact = true;
		isActive = false;
		isPassive = (pmdSuccess > randomNumber());
		explosionProbability = rocketBodyExplosionProbability;
		lifetime = 10;
		break;
	case 1:
		isIntact = true;
		isActive = true;
		explosionProbability = satelliteExplosionProbability;
		avoidanceSucess = 1;

		lifetime = 50 * 365.25;
		break;
		//TODO - update lifetime to be object dependent
	case 2:
		isIntact = false;
		isActive = false;
		explosionProbability = 0.;
		lifetime = 0;
		break;
	}
}

DebrisObject::DebrisObject(string TLE1, string TLE2, string TLE3) : DebrisObject(TLE2, TLE3)
{
	strcpy(name, TLE1.c_str());
}

DebrisObject::DebrisObject(string TLE2, string TLE3)
{
	int epochYear;
	double meanMotion, semiMajorAxis, eccentricity, inclination, rightAscension, argPerigee, init_meanAnomaly, epochDay;

	objectID = ++objectSEQ;
	sourceID = objectID;
	parentID = 0;
	noradID = stoi(TLE3.substr(2, 5));

	// Convert to days since 1957-OCT-04
	epochYear = stoi(TLE2.substr(18, 2));
	epochDay = stod(TLE2.substr(20, 12));
	
	//Handle y2k
	if (epochYear < 57)
		epochYear += 2000;
	else
		epochYear += 1900;

	int mon, day, hr, min;
	double sec;

	SGP4Funcs::days2mdhms(epochYear, epochDay, mon, day, hr, min, sec);
	initEpoch = DateToEpoch(epochYear, mon, day, hr, min, sec);
	currEpoch = initEpoch;

	bStar = stod(TLE2.substr(53, 1) + "0." + TLE2.substr(54, 5) + "e" + TLE2.substr(59, 2));

	inclination = DegToRad(stod(TLE3.substr(8, 8)));
	rightAscension = DegToRad(stod(TLE3.substr(17, 8)));
	eccentricity = stod("0." + TLE3.substr(26, 7));
	argPerigee = DegToRad(stod(TLE3.substr(34, 8)));
	init_meanAnomaly = DegToRad(stod(TLE3.substr(43, 8)));
	meanMotion = stod(TLE3.substr(52, 11)) * Tau / secondsDay;
	semiMajorAxis = cbrt(muGravity / (meanMotion * meanMotion));

	elements = OrbitalElements(semiMajorAxis, eccentricity, inclination, rightAscension, argPerigee, init_meanAnomaly);

	mass = 1;
	meanAnomalyEpoch = init_meanAnomaly;
	nFrag = 1;
	objectType = 2;
	positionSync = velocitySync = false;
	periodSync = false;
	coefficientDrag = 2.2;
	isIntact = false;
	isActive = false;
	explosionProbability = 0.;
}

DebrisObject::~DebrisObject()
{
}

void DebrisObject::RegenerateID()
{
	objectID = ++objectSEQ;
}

long DebrisObject::GetID()
{
	return objectID;
}

long DebrisObject::GetSourceID()
{
	return sourceID;
}

long DebrisObject::GetParentID()
{
	return parentID;
}

int DebrisObject::GetConstellationID()
{
	return constellationId;
}

int DebrisObject::GetNoradID()
{
	if (noradID != -1)
		return noradID;
	else
		return objectID;
}

int DebrisObject::GetType()
{
	return objectType;
}

int DebrisObject::GetSourceType()
{
	return sourceType;
}

int DebrisObject::GetSourceEvent()
{
	return sourceEvent;
}

void DebrisObject::RemoveObject(int removeType, double epoch) // (0, 1, 2) = (Decay, Explosion, Collision) respectively.
{
	removeEpoch = epoch;
	removeEvent = removeType;
	isIntact = false;
}

int DebrisObject::GetRemoveEvent()
{
	return removeEvent;
}

double DebrisObject::GetRemoveEpoch()
{
	return removeEpoch;
}

void DebrisObject::SetName(string init_name)
{
	strcpy(name, init_name.c_str());
}

string DebrisObject::GetName()
{
	string tempname(name);
	if (tempname.size() > 0)
		return tempname;
	else
		return to_string(objectID);
}

int DebrisObject::GetNFrag()
{
	return nFrag;
}

double DebrisObject::GetInitEpoch()
{
	return initEpoch;
}

double DebrisObject::GetEpoch()
{
	return currEpoch;
}

void DebrisObject::UpdateOrbitalElements(vector3D deltaV)
{
	velocity.addVector(deltaV);
	elements.SetOrbitalElements(position, velocity);
	velocitySync = positionSync = true;
	periodSync = false;
}

void DebrisObject::UpdateOrbitalElements(OrbitalElements newElements)
{
	elements = OrbitalElements(newElements);
	positionSync = velocitySync = false;
	periodSync = false;
}

void DebrisObject::UpdateOrbitalElements(vector3D position, vector3D velocity)
{
	elements = OrbitalElements(position, velocity);
	positionSync = velocitySync = false;
	periodSync = false;
}


double DebrisObject::GetApogee()
{
	return elements.GetApogee();
}

double DebrisObject::GetCDrag()
{
	return coefficientDrag;
}

double DebrisObject::GetBStar()
{
	return bStar;
}

double DebrisObject::GetLaunchCycle()
{
	return launchCycle;
}

double DebrisObject::GetAvoidanceSuccess()
{
	if (isActive)
		return avoidanceSucess;
	else
		return 0.;
}

double DebrisObject::GetExplosionProbability()
{
	double modifier;
	if (isActive)
		//ToDo - implement variation in explosion probability based on age
		modifier = 1;
	else if (isIntact && !isPassive)
		// ToDo - implement modifier based on passivation
		modifier = 1;
	else
		modifier = 0.;

	return explosionProbability * modifier;
}

double DebrisObject::GetCollisionProbability()
{
	return collisionProbability;
}

bool DebrisObject::IsIntact()
{
	return isIntact;
}

bool DebrisObject::IsActive()
{
	return isActive;
}

bool DebrisObject::IsPassive()
{
	return isPassive;
}

vector3D DebrisObject::GetVelocity()
{
	if (!velocitySync)
	{
		velocity = vector3D(elements.GetVelocity());
		velocitySync = true;
	}
	return velocity;
}

void DebrisObject::SetVelocity(double vX, double vY, double vZ)
{
	velocity = vector3D(vX, vY, vZ);
	elements.SetOrbitalElements(position, velocity);
	velocitySync = positionSync = true;
	periodSync = false;
}

void DebrisObject::SetVelocity(vector3D inputVelocity)
{
	velocity = vector3D(inputVelocity);
	elements.SetOrbitalElements(position, velocity);
	velocitySync = positionSync = true;
	periodSync = false;
}

vector3D DebrisObject::GetPosition()
{
	if (!positionSync)
	{
		position = vector3D(elements.GetPosition());
		positionSync = true;
	}

	return position;
}

vector<double> DebrisObject::GetStateVector()
{
	vector<double> stateVector(6);

	if (!isIntact)
	{
		stateVector = { NAN, NAN, NAN, NAN, NAN, NAN };
	}
	else
	{
		if (!positionSync)
		{
			position = vector3D(elements.GetPosition());
			positionSync = true;
		}
		if (!velocitySync)
		{
			velocity = vector3D(elements.GetVelocity());
			velocitySync = true;
		}
		stateVector = { position.x, position.y, position.z, velocity.x, velocity.y, velocity.z };
	}
	return stateVector;
}

vector3D DebrisObject::GetNormalVector()
{
	return elements.GetNormalVector();
}

void DebrisObject::SetPosition(double X, double Y, double Z) // Not Safe - invalid logic
{
	position = vector3D(X, Y, Z);
	elements.SetOrbitalElements(position, velocity);
	velocitySync = positionSync = true;
	periodSync = false;
}

void DebrisObject::SetPosition(vector3D inputPosition) // Not Safe - invalid logic
{
	position = vector3D(inputPosition);
	elements.SetOrbitalElements(position, velocity);
	meanAnomalyEpoch = elements.GetMeanAnomaly();
	velocitySync = positionSync = true;
	periodSync = false;
}


void DebrisObject::SetStateVectors(vector3D inputPosition, vector3D inputVelocity)
{
	position = vector3D(inputPosition);
	velocity = vector3D(inputVelocity);
	elements.SetOrbitalElements(position, velocity);
	meanAnomalyEpoch = elements.GetMeanAnomaly();
	velocitySync = positionSync = true;
	periodSync = false;
}

void DebrisObject::SetStateVectors(double X, double Y, double Z, double vX, double vY, double vZ)
{
	position = vector3D(X, Y, Z);
	velocity = vector3D(vX, vY, vZ);
	elements.SetOrbitalElements(position, velocity);
	velocitySync = positionSync = true;
	periodSync = false;
}

void DebrisObject::CalculateMassFromArea()
{
	mass = area / areaToMass;
}


void DebrisObject::CalculateAreaFromMass()
{
	area = mass * areaToMass;
}

void DebrisObject::CalculateAreaToMass()
{
	areaToMass = area / mass;
}

double DebrisObject::GetMass()
{
	return mass;
}

double DebrisObject::GetLength()
{
	return length;
}

double DebrisObject::GetArea()
{
	return area;
}

double DebrisObject::GetAreaToMass()
{
	return areaToMass;
}

double DebrisObject::GetRadius()
{
	return radius;
}

double DebrisObject::GetPeriod()
{
	if (!periodSync)
	{
		period = elements.CalculatePeriod();
		periodSync = true;
	}
	return period;
}

double DebrisObject::GetEpochAnomaly()
{
	return meanAnomalyEpoch;
}

double DebrisObject::GetPerigee()
{
	return elements.GetPerigee();
}


void DebrisObject::SetSourceID(long ID)
{
	sourceID = ID;
}

void DebrisObject::SetCentralBody(int c)
{
	elements.centralBody = c;
}

void DebrisObject::SetParentID(long ID)
{
	parentID = ID;
}

void DebrisObject::SetCDrag(double cDrag)
{
	coefficientDrag = cDrag;
}

void DebrisObject::SetBStar(double b)
{
	bStar = b;
}

void DebrisObject::SetInitEpoch(double epoch)
{
	initEpoch = epoch;
}

void DebrisObject::SetEpoch(double epoch)
{
	currEpoch = epoch;
}

void DebrisObject::UpdateEpoch(double epochStep)
{
	currEpoch += epochStep;

	// Determine if spacecraft reaches end of life
	if (currEpoch >= initEpoch + lifetime)
	{
		isActive = false;
		bStar = NAN;
		isPassive = (pmdSuccess > randomNumber());
		//TODO - include PMD disposal orbit
	}
}

void DebrisObject::UpdateCollisionProbability(double probability)
{
	double inverseProbability;

	inverseProbability = 1.0 - collisionProbability;
	inverseProbability = inverseProbability * (1.0 - probability);

	collisionProbability = 1.0 - inverseProbability;
}

void DebrisObject::SetRadius(double radii)
{
	radius = radii;
}

void DebrisObject::SetArea(double xsection)
{
	area = xsection;
	CalculateAreaToMass();
}

void DebrisObject::SetMass(double newMass)
{
	mass = newMass;
}

void DebrisObject::SetNFrag(int n)
{
	nFrag = 1;
}

void DebrisObject::SetNewObjectID()
{
	objectID = ++objectSEQ;
}

void DebrisObject::SetConstellationID(int id)
{
	constellationId = id;
}

void DebrisObject::SetLaunchCycle(double cycle)
{
	launchCycle = cycle;
}

OrbitalAnomalies DebrisObject::GetAnomalies()
{
	return elements.GetAnomalies();
}

OrbitalElements& DebrisObject::GetElements()
{
	return elements;
}

bool DebrisObject::SGP4Initialised()
{
	return sgp4Initialised;
}

elsetrec & DebrisObject::GetSGP4SatRec()
{
	if (!sgp4Initialised)
	{
		sgp4Sat = elsetrec();
		sgp4Initialised = true;
	}
	return sgp4Sat;
}

void DebrisObject::RemoveNFrag()
{
	--nFrag;
}

void DebrisObject::UpdateRAAN(double rightAscension)
{
	elements.SetRightAscension(rightAscension);
	positionSync = velocitySync = false;
}

void DebrisObject::UpdateArgP(double argPerigee)
{
	elements.SetArgPerigee(argPerigee);
	positionSync = velocitySync = false;
}

void DebrisObject::RandomiseMeanAnomaly()
{
	double M = randomNumberTau();
	SetMeanAnomaly(M);
	meanAnomalyEpoch = M;
}

void DebrisObject::SetMeanAnomaly(double M)
{
	elements.SetMeanAnomaly(M);
	positionSync = velocitySync = false;
}

void DebrisObject::SetTrueAnomaly(double v)
{
	elements.SetTrueAnomaly(v);
	positionSync = velocitySync = false;
}

void DebrisObject::SetEccentricAnomaly(double E)
{
	elements.SetEccentricAnomaly(E);
	positionSync = velocitySync = false;
}

DebrisObject CopyDebrisObject(DebrisObject & object)
{
	DebrisObject newObject = object;
	newObject.RegenerateID();
	return newObject;
}

bool CompareInitEpochs(DebrisObject objectA, DebrisObject objectB)
{
	return objectA.GetInitEpoch() < objectB.GetInitEpoch();
}
