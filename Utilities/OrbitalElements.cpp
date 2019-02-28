#include "stdafx.h"
#include "OrbitalElements.h"

// Elements functions
OrbitalElements::OrbitalElements() {}

OrbitalElements::OrbitalElements(double a, double e, double i, double ra, double ap, double meanAnomaly)
{
	semiMajorAxis = a;
	eccentricity = e;
	inclination = i;
	rightAscension = ra;
	argPerigee = ap;
	anomalies.SetMeanAnomaly(meanAnomaly);
	anomaliesSynced = true;
}


OrbitalElements::OrbitalElements(vector3D &position, vector3D &velocity)
{
	double trueAnomaly;

	vector3D angularMomentum = position.VectorCrossProduct(velocity);
	vector3D node(-1 * angularMomentum.y, angularMomentum.x, 0.0);
	vector3D eccentricityVector = CalculateEccentricityVector(position, velocity, angularMomentum);

	semiMajorAxis = 1 / (2 / position.vectorNorm() - velocity.vectorNorm2() / muGravity);

	eccentricity = eccentricityVector.vectorNorm();

	inclination = acos(angularMomentum.z / angularMomentum.vectorNorm()); // Returns on [0,PI] range which is correct

	rightAscension = acos(node.x / node.vectorNorm()); // Returns on [0,PI] range which is incorrect
	if (node.y < 0)
		rightAscension = Tau - rightAscension; // Corrected to [0,Tau] range

	argPerigee = acos(node.VectorDotProduct(eccentricityVector) / (node.vectorNorm() * eccentricity)); // Returns on [0,PI] range which is incorrect
	if (eccentricityVector.z < 0)
		argPerigee = Tau - argPerigee; // Corrected to [0,Tau] range

	trueAnomaly = acos(eccentricityVector.VectorDotProduct(position) / (eccentricity * position.vectorNorm())); // Returns on [0,PI] range which is incorrect
	if (position.VectorDotProduct(velocity) < 0)
		trueAnomaly = Tau - trueAnomaly; // Corrected to [0,Tau] range

	anomalies.SetTrueAnomaly(trueAnomaly);
	anomaliesSynced = true;
}

vector3D OrbitalElements::CalculateEccentricityVector(vector3D& position, vector3D& velocity, vector3D &angularMomentum)
{
	vector3D eccentricityVector;
	double velocitySquared = velocity.vectorNorm2();
	double radialPosition = position.vectorNorm();
	eccentricityVector = (position * (velocitySquared - muGravity / radialPosition) - velocity * (position.VectorDotProduct(velocity))) / muGravity;
	return eccentricityVector;
}

double OrbitalElements::GetTrueAnomaly()
{
	return anomalies.GetTrueAnomaly(eccentricity);
}

double OrbitalElements::GetMeanAnomaly()
{
	return anomalies.GetMeanAnomaly(eccentricity);
}

double OrbitalElements::GetEccentricAnomaly()
{
	return anomalies.GetEccentricAnomaly(eccentricity);
}

void OrbitalElements::SetOrbitalElements(double a, double e, double i, double ra, double ap)
{
	semiMajorAxis = a;
	eccentricity = e;
	inclination = i;
	rightAscension = ra;
	argPerigee = ap;
	anomaliesSynced = false;
}

void OrbitalElements::SetOrbitalElements(vector3D & position, vector3D & velocity)
{
	double trueAnomaly;

	vector3D angularMomentum = position.VectorCrossProduct(velocity);
	vector3D node(-1 * angularMomentum.y, angularMomentum.x, 0.0);
	vector3D eccentricityVector = CalculateEccentricityVector(position, velocity, angularMomentum);

	semiMajorAxis = 1 / (2 / position.vectorNorm() - velocity.vectorNorm2() / muGravity);

	eccentricity = eccentricityVector.vectorNorm();

	inclination = acos(angularMomentum.z / angularMomentum.vectorNorm()); // Returns on [0,PI] range which is correct

	rightAscension = acos(node.x / node.vectorNorm()); // Returns on [0,PI] range which is incorrect
	if (node.y < 0)
		rightAscension = Tau - rightAscension; // Corrected to [0,Tau] range

	argPerigee = acos(node.VectorDotProduct(eccentricityVector) / (node.vectorNorm() * eccentricity)); // Returns on [0,PI] range which is incorrect
	if (eccentricityVector.z < 0)
		argPerigee = Tau - argPerigee; // Corrected to [0,Tau] range

	trueAnomaly = acos(eccentricityVector.VectorDotProduct(position) / (eccentricity * position.vectorNorm())); // Returns on [0,PI] range which is incorrect
	if (position.VectorDotProduct(velocity) < 0)
		trueAnomaly = Tau - trueAnomaly; // Corrected to [0,Tau] range

	anomalies.SetTrueAnomaly(trueAnomaly);
	anomaliesSynced = true;
}

void OrbitalElements::SetRightAscension(double init_rightAscension)
{
	rightAscension = init_rightAscension;
}

void OrbitalElements::SetArgPerigee(double init_argPerigee)
{
	argPerigee = init_argPerigee;
}

double OrbitalElements::GetRadialPosition()
{
	double radius;
	double trueAnomaly = anomalies.GetTrueAnomaly(eccentricity);
		
	radius = semiMajorAxis * (1 - eccentricity * eccentricity) / (1 + eccentricity * cos(trueAnomaly));

	return radius;
}

double OrbitalElements::GetRadialPosition(double trueAnomaly)
{
	double radius;

	radius = semiMajorAxis * (1 - eccentricity * eccentricity) / (1 + eccentricity * cos(trueAnomaly));

	return radius;
}

double OrbitalElements::CalculatePeriod()
{
	return Tau * sqrt(semiMajorAxis * semiMajorAxis * semiMajorAxis / muGravity);
}

vector3D OrbitalElements::GetPosition()
{
	double radius, trueAnomaly, x, y, z, U;
	double sinRA, cosRA, sinI, cosI, sinU, cosU;
	
	trueAnomaly = anomalies.GetTrueAnomaly(eccentricity);
	radius = GetRadialPosition();
	U = argPerigee + trueAnomaly;

	sinRA = sin(rightAscension);
	cosRA = cos(rightAscension);
	sinI = sin(inclination);
	cosI = cos(inclination);
	sinU = sin(U);
	cosU = cos(U);

	x = radius * (cosRA * cosU - sinRA * sinU * cosI);
	y = radius * (sinRA * cosU + cosRA * sinU * cosI);
	z = radius * (sinI * sinU);
	
	return vector3D(x, y, z);
}

vector3D OrbitalElements::GetVelocity()
{
	double radius, trueAnomaly, p, h, A, B, x, y, z, vX, vY, vZ;
	double sinRA, cosRA, sinI, cosI, sinU, cosU;

	radius = GetRadialPosition();
	trueAnomaly = anomalies.GetTrueAnomaly(eccentricity);

	sinRA = sin(rightAscension);
	cosRA = cos(rightAscension);
	sinI = sin(inclination);
	cosI = cos(inclination);
	sinU = sin(argPerigee + trueAnomaly);
	cosU = cos(argPerigee + trueAnomaly);

	p = semiMajorAxis * (1 - eccentricity * eccentricity);
	h = sqrt(muGravity * p);

	A = sin(trueAnomaly) * h * eccentricity / p;
	B = h / radius;

	x = (cosRA * cosU - sinRA * sinU * cosI);
	y = (sinRA * cosU + cosRA * sinU * cosI);
	z = (sinI * sinU);

	vX = x * A - B * (cosRA * sinU + sinRA * cosU * cosI);
	vY = y * A - B * (sinRA * sinU - cosRA * cosU * cosI);
	vZ = z * A + B * (cosU * sinI);

	return vector3D(vX, vY, vZ);
}

vector3D OrbitalElements::GetNormalVector()
{
	double i, j, k;
	i = sin(rightAscension) * sin(inclination);
	j = cos(rightAscension) * sin(inclination);
	k = cos(inclination);
	return vector3D(i, j, k);
}

double OrbitalElements::GetPerigee()
{
	return semiMajorAxis * (1 - eccentricity);
}

double OrbitalElements::GetApogee()
{
	return semiMajorAxis * (1 + eccentricity);
}

OrbitalAnomalies OrbitalElements::GetAnomalies()
{
	return anomalies;
}

void OrbitalElements::SetMeanAnomaly(double M)
{
	anomalies.SetMeanAnomaly(M);
}

void OrbitalElements::SetTrueAnomaly(double v)
{
	anomalies.SetTrueAnomaly(v);
}

void OrbitalElements::SetEccentricAnomaly(double E)
{
	anomalies.SetEccentricAnomaly(E);
}