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
	double GetMeanAnomaly();
	double GetEccentricAnomaly();
	void UpdateOrbitalElements(double deltaSMA = 0, double deltaEcc = 0, double deltaInc = 0, double deltaRA = 0, double deltaAP = 0);
	void SetOrbitalElements(double a, double e, double i, double ra, double ap);
	void SetOrbitalElements(vector3D &position, vector3D &velocity);
	void SetRightAscension(double init_rightAscension);
	void SetArgPerigee(double init_argPerigee);

	vector3D GetPosition();
	vector3D GetVelocity();
	vector3D GetNormalVector();
	double GetPerigee();
	double GetApogee();
	double GetRadialPosition();
	double GetRadialPosition(double trueAnomaly);
	double CalculatePeriod();
	double GetMeanMotion();

	OrbitalAnomalies GetAnomalies();
	void SetMeanAnomaly(double M);
	void SetTrueAnomaly(double v);
	void SetEccentricAnomaly(double E);


private:
	vector3D CalculateEccentricityVector(vector3D &position, vector3D &velocity, vector3D &angularMomentum);
};