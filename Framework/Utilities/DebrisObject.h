#pragma once

class DebrisObject
{
public:
	OrbitalElements elements; // semi-major axis, eccentricity, inclination, right ascension of ascending node, arguement of perigee
	OrbitalAnomalies anomalies; // mean anomaly, eccentric anomaly, true anomaly
	double meanAnomalyEpoch;
	//protected:
	float radius, mass, length;
	long parentID, sourceID, objectID;
	int sourceType; // (0, 1, 2) = (Launch, Explosion, Collision) respectively.

public:
	DebrisObject();
	DebrisObject(float init_radius, float init_mass, float init_length, double semiMajorAxis, double eccentricity, double inclination, double rightAscension,
		double argPerigee, double init_meanAnomaly);
	~DebrisObject();
	void UpdateOrbitalElements(double deltaV);
	vector3D GetVelocity();

};

