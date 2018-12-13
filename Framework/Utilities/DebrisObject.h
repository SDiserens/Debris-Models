#pragma once


class DebrisObject
{
public:
	OrbitalElements elements; // semi-major axis, eccentricity, inclination, right ascension of ascending node, arguement of perigee
	OrbitalAnomalies anomalies; // mean anomaly, eccentric anomaly, true anomaly
	
protected:
	static int objectSEQ;
	double meanAnomalyEpoch, radius, mass, length, area, areaToMass, velocityNorm;
	long parentID, sourceID, objectID;
	int sourceType; // (0, 1, 2) = (Launch, Explosion, Collision) respectively.
	vector3D velocity, position;

public:
	DebrisObject();
	DebrisObject(double init_radius, double init_mass, double init_length, double semiMajorAxis, double eccentricity, double inclination, double rightAscension,
		double argPerigee, double init_meanAnomaly);
	~DebrisObject();

	double GetMass();
	double GetLength();
	double GetArea();
	double GetAreaToMass();
	vector3D GetVelocity();
	vector3D GetPosition();
	
	void SetVelocity(double vX, double vY, double vZ);
	void SetVelocity(vector3D inputVelocity);
	void SetPosition(double X, double Y, double Z);
	void SetPosition(vector3D inputPosition);
	void UpdateOrbitalElements(vector3D deltaV);

protected:
	void CalculateMassFromArea();
	void CalculateAreaFromMass();
	void CalculateAreaToMass();
};

