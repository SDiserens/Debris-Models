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
	CUDA_CALLABLE_MEMBER double GetMeanAnomaly();
	double GetEccentricAnomaly();
	void UpdateOrbitalElements(double deltaSMA = 0, double deltaEcc = 0, double deltaInc = 0, double deltaRA = 0, double deltaAP = 0);
	void SetOrbitalElements(double a, double e, double i, double ra, double ap);
	void SetOrbitalElements(vector3D &position, vector3D &velocity);
	void SetRightAscension(double init_rightAscension);
	void SetArgPerigee(double init_argPerigee);

	CUDA_CALLABLE_MEMBER vector3D GetPosition();
	CUDA_CALLABLE_MEMBER vector3D GetVelocity();
	CUDA_CALLABLE_MEMBER vector3D GetNormalVector();
	double GetPerigee();
	double GetApogee();
	CUDA_CALLABLE_MEMBER double GetRadialPosition();
	CUDA_CALLABLE_MEMBER double GetRadialPosition(double trueAnomaly);
	CUDA_CALLABLE_MEMBER double CalculatePeriod();
	double GetMeanMotion();

	CUDA_CALLABLE_MEMBER OrbitalAnomalies GetAnomalies();
	void SetMeanAnomaly(double M);
	void SetTrueAnomaly(double v);
	void SetEccentricAnomaly(double E);


private:
	vector3D CalculateEccentricityVector(vector3D &position, vector3D &velocity, vector3D &angularMomentum);
};