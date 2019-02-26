#pragma once
class OrbitalElements
{
public:
	double semiMajorAxis, eccentricity, inclination, rightAscension, argPerigee;
	OrbitalAnomalies anomalies; // mean anomaly, eccentric anomaly, true anomaly
	bool anomaliesSynced;

public:
	OrbitalElements();
	OrbitalElements(double a, double e, double i, double ra, double ap, double meanAnomaly);
	OrbitalElements(vector3D &position, vector3D &velocity);
	double GetTrueAnomaly();
	void SetOrbitalElements(double a, double e, double i, double ra, double ap);
	void SetOrbitalElements(vector3D &position, vector3D &velocity);
	void SetRightAscension(double init_rightAscension);
	void SetArgPerigee(double init_argPerigee);

	vector3D GetPostion();
	vector3D GetVelocity();
	double GetRadialPosition();
	double GetRadialPosition(double trueAnomaly);
	double CalculatePeriod();

	OrbitalAnomalies GetAnomalies();
	void SetMeanAnomaly(double M);
	void SetTrueAnomaly(double v);
	void SetEccentricAnomaly(double E);


private:
	vector3D CalculateEccentricityVector(vector3D &position, vector3D &velocity, vector3D &angularMomentum);
};