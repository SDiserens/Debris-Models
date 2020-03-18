#pragma once

#include "Modules\Propagators\SGP4\SGP4Code\SGP4.h"

static double rocketBodyExplosionProbability = 0;
static double satelliteExplosionProbability = 0;
static double pmdSuccess = 0;

class DebrisObject
{
protected:
	OrbitalElements elements; // semi-major axis, eccentricity, inclination, right ascension of ascending node, arguement of perigee, anomalies
	static int objectSEQ;
	double meanAnomalyEpoch, radius, mass, length, area, areaToMass, removeEpoch, period, coefficientDrag, initEpoch, lifetime, bStar, currEpoch, avoidanceSucess=0, explosionProbability=0;
	char name[100];
	long parentID, sourceID, objectID;
	int sourceEvent; // (0, 1, 2) = (Launch, Explosion, Collision) respectively.
	int sourceType, objectType; // (0, 1, 2) = (UpperStage, Spacecraft, Debris) respectively. //TODO distinguish between active/inactive?
	int removeEvent; // (0, 1, 2) = (Decay, Explosion, Collision) respectively.
	int nFrag, noradID;
	vector3D velocity, position;
	bool positionSync, velocitySync, periodSync, isActive, isIntact, isPassive=false;
	elsetrec sgp4Sat;
	bool sgp4Initialised = false;

public:
	DebrisObject();
	DebrisObject(double init_radius, double init_mass, double init_length, double semiMajorAxis, double eccentricity, double inclination, double rightAscension,
		double argPerigee, double init_meanAnomaly, int type);
	DebrisObject(string TLE1, string TLE2, string TLE3);
	DebrisObject(string TLE2, string TLE3);
	CUDA_CALLABLE_MEMBER ~DebrisObject();
	void RegenerateID();

	long GetID();
	long GetSourceID();
	long GetParentID();
	int GetNoradID();
	int GetType();
	int GetSourceType();
	int GetSourceEvent();
	void RemoveObject(int removeType, double epoch); // (0, 1, 2) = (Decay, Explosion, Collision) respectively.
	
	void SetName(string init_name);
	string GetName();
	int GetNFrag();
	double GetInitEpoch();
	double GetEpoch();
	CUDA_CALLABLE_MEMBER double GetMass();
	double GetLength();
	double GetArea();
	double GetAreaToMass();
	double GetRadius();
	CUDA_CALLABLE_MEMBER double GetPeriod();
	double GetEpochAnomaly();
	double GetPerigee();
	double GetApogee();
	double GetCDrag();
	double GetBStar();
	double GetAvoidanceSuccess();
	double GetExplosionProbability();
	bool IsIntact();
	bool IsActive();
	CUDA_CALLABLE_MEMBER vector3D GetVelocity();
	vector3D GetPosition();
	vector<double> GetStateVector();
	vector3D GetNormalVector();

	CUDA_CALLABLE_MEMBER OrbitalAnomalies GetAnomalies();
	CUDA_CALLABLE_MEMBER OrbitalElements& GetElements();

	bool SGP4Initialised();
	elsetrec& GetSGP4SatRec();

	void UpdateRAAN(double rightAscension);
	void UpdateArgP(double argPerigee);
	void RandomiseMeanAnomaly();
	void SetMeanAnomaly(double M);
	void SetTrueAnomaly(double v);
	void SetEccentricAnomaly(double E);
	void UpdateOrbitalElements(vector3D deltaV);
	void UpdateOrbitalElements(OrbitalElements newElements);
	void UpdateOrbitalElements(vector3D position, vector3D velocity);
	void UpdateEpoch(double epochStep);

	void SetSourceID(long ID);
	void SetParentID(long ID);
	void SetCDrag(double cDrag);
	void SetBStar(double bStar);
	void SetInitEpoch(double epoch);
	void SetEpoch(double epoch);
	void SetRadius(double radii);

	void SetVelocity(double vX, double vY, double vZ);
	void SetVelocity(vector3D inputVelocity);
	void SetPosition(double X, double Y, double Z); //Risky
	void SetPosition(vector3D inputPosition); //Risky
	void SetStateVectors(vector3D inputPosition, vector3D inputVelocity);
	void SetStateVectors(double X, double Y, double Z, double vX, double vY, double vZ);

protected:
	void CalculateMassFromArea();
	void CalculateAreaFromMass();
	void CalculateAreaToMass();
};


DebrisObject CopyDebrisObject(DebrisObject & object);

bool CompareInitEpochs(DebrisObject objectA, DebrisObject objectB);