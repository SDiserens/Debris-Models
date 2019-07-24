#include "stdafx.h"

int DebrisObject::objectSEQ = 0;

DebrisObject::DebrisObject() 
{
	positionSync = velocitySync = false;
	periodSync = false;
}

DebrisObject::DebrisObject(double init_radius, double init_mass, double init_length, double semiMajorAxis, double eccentricity, double inclination,
	double rightAscension, double argPerigee, double init_meanAnomaly, int type)
{
	objectID = ++objectSEQ; // This should be generating a unique ID per object incl. when threading (needs revision for multi-thread)
	radius = init_radius;
	mass = init_mass;
	length = init_length;
	elements = OrbitalElements(semiMajorAxis, eccentricity, inclination, rightAscension, argPerigee, init_meanAnomaly);
	meanAnomalyEpoch = init_meanAnomaly;
	nFrag = 1;
	objectType = type;
	positionSync = velocitySync = false;
	periodSync = false;
	coefficientDrag = 2.2;
	bStar = NAN;
}

DebrisObject::DebrisObject(string TLE1, string TLE2, string TLE3) : DebrisObject(TLE2, TLE3)
{
	name = TLE1;
}

DebrisObject::DebrisObject(string TLE2, string TLE3)
{

	double meanMotion, semiMajorAxis, eccentricity, inclination, rightAscension, argPerigee, init_meanAnomaly;
	objectID = ++objectSEQ;

	bStar = stod(TLE2.substr(53, 1) + "0." + TLE2.substr(54, 5) + "e" + TLE2.substr(59, 2));

	meanMotion = stod(TLE3.substr(52, 11));
	semiMajorAxis = cbrt(muGravity / (meanMotion * meanMotion));
	inclination = DegToRad(stod(TLE3.substr(8, 8)));
	rightAscension = DegToRad(stod(TLE3.substr(17, 8)));
	eccentricity = stod("0." + TLE3.substr(26, 7));
	argPerigee = DegToRad(stod(TLE3.substr(34, 8)));
	init_meanAnomaly = DegToRad(stod(TLE3.substr(43, 8)));

	elements = OrbitalElements(semiMajorAxis, eccentricity, inclination, rightAscension, argPerigee, init_meanAnomaly);

	meanAnomalyEpoch = init_meanAnomaly;
	nFrag = 1;
	objectType = 2;
	positionSync = velocitySync = false;
	periodSync = false;
	coefficientDrag = 2.2;
}

DebrisObject::~DebrisObject()
{
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
}

void DebrisObject::SetName(string init_name)
{
	name = init_name;
}

string DebrisObject::GetName()
{
	return name;
}

int DebrisObject::GetNFrag()
{
	return nFrag;
}

double DebrisObject::GetInitEpoch()
{
	return initEpoch;
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
	vector<double> stateVector{ position.x, position.y, position.z, velocity.x, velocity.y, velocity.z };
	
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
	velocitySync = positionSync = true;
	periodSync = false;
}


void DebrisObject::SetStateVectors(vector3D inputPosition, vector3D inputVelocity)
{
	position = vector3D(inputPosition);
	velocity = vector3D(inputVelocity);
	elements.SetOrbitalElements(position, velocity);
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

void DebrisObject::SetParentID(long ID)
{
	parentID = ID;
}

void DebrisObject::SetCDrag(double cDrag)
{
	coefficientDrag = cDrag;
}

void DebrisObject::SetBStar(double bStar)
{
	bStar = bStar;
}

void DebrisObject::SetInitEpoch(double epoch)
{
	initEpoch = epoch;
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
