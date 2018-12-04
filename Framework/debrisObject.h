#pragma once
using namespace std;
#include <vector>

class DebrisObject
{
public:
	DebrisObject(float init_radius, float init_mass, float init_length, double semiMajorAxis, double eccentricity, double inclination, double rightAscension,
		double argPerigee, double init_meanAnomaly);
	~DebrisObject();
	void UpdateOrbitalElements(double deltaV);
public:
	std::vector<double> elements; // semi-major axis, eccentricity, inclination, right ascension of ascending node, arguement of perigee
	std::vector<double> anomalies; // mean anomaly, eccentric anomaly, true anomaly
	double meanAnomalyEpoch;
//protected:
	float radius, mass, length;
	long parentID, sourceID, objectID;
};

