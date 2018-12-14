#include "stdafx.h"
#include "OrbitalElements.h"

// Elements functions
OrbitalElements::OrbitalElements() {}

OrbitalElements::OrbitalElements(double a, double e, double i, double ra, double ap)
{
	semiMajorAxis = a;
	eccentricity = e;
	inclination = i;
	rightAscension = ra;
	argPerigee = ap;
}


OrbitalElements::OrbitalElements(vector3D &position, vector3D &velocity)
{
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
	double tempAnomaly = trueAnomaly;
	trueAnomaly = NULL;
	return tempAnomaly;
}
