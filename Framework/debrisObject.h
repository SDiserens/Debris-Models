#pragma once
class DebrisObject
{
public:
	DebrisObject(float init_radius, float init_mass, float init_length, double semiMajorAxis, double eccentricity, double inclination, double rightAscension,
		double argPerigee, double init_meanAnomaly);
	~DebrisObject();
	std::vector<double> elements; // semi-major axis, eccentricity, inclination, right ascension of ascending node, arguement of perigee
	std::vector<double> anomalies; // mean anomaly, eccentric anomaly, true anomaly
	float radius, mass, length, meanAnomalyEpoch;
	long parentID, sourceID, objectID;
};

