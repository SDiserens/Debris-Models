#pragma once

#include "Modules\Propagators\SGP4\SGP4Code\SGP4.h"

static double rocketBodyExplosionProbability = 0;
static double satelliteExplosionProbability = 0;
static double pmdSuccess = 0;
static double manoeuvreThreshold = 0.0001;

class DebrisObject
{
protected:
	OrbitalElements elements; // semi-major axis, eccentricity, inclination, right ascension of ascending node, arguement of perigee, anomalies

	double meanAnomalyEpoch, radius, mass, length, area, areaToMass, removeEpoch, period, coefficientDrag, initEpoch, lifetime, bStar, currEpoch, launchCycle;
	double avoidanceSucess = 0, explosionProbability = 0, collisionProbability=0;
	char name[100];
	long parentID, sourceID, objectID;
	int constellationId=-1;
	int sourceEvent; // (0, 1, 2) = (Launch, Explosion, Collision) respectively.
	int sourceType, objectType; // (0, 1, 2) = (UpperStage, Spacecraft, Debris) respectively. 
	int removeEvent; // (0, 1, 2) = (Decay, Explosion, Collision) respectively.
	int nFrag, noradID;
	vector3D velocity, position;
	bool positionSync, velocitySync, periodSync, isActive, isIntact, isPassive = false;
	elsetrec sgp4Sat;
	bool sgp4Initialised = false;

public:
	CUDA_CALLABLE_MEMBER DebrisObject();
	DebrisObject(double init_radius, double init_mass, double init_length, double semiMajorAxis, double eccentricity, double inclination, double rightAscension,
		double argPerigee, double init_meanAnomaly, int type);
	DebrisObject(string TLE1, string TLE2, string TLE3);
	DebrisObject(string TLE2, string TLE3);
	CUDA_CALLABLE_MEMBER ~DebrisObject();
	void RegenerateID(long ID);

	CUDA_CALLABLE_MEMBER long GetID();
	long GetSourceID();
	long GetParentID();
	int GetConstellationID();
	int GetNoradID();
	int GetType();
	int GetSourceType();
	int GetSourceEvent();
	void RemoveObject(int removeType, double epoch); // (0, 1, 2) = (Decay, Explosion, Collision) respectively.
	int GetRemoveEvent();
	double GetRemoveEpoch();

	void SetName(string init_name);
	string GetName();
	int GetNFrag();
	double GetInitEpoch();
	double GetEpoch();
	CUDA_CALLABLE_MEMBER double GetMass();
	double GetLength();
	double GetArea();
	double GetAreaToMass();
	CUDA_CALLABLE_MEMBER double GetRadius();
	CUDA_CALLABLE_MEMBER double GetPeriod();
	CUDA_CALLABLE_MEMBER double GetEpochAnomaly();
	double GetPerigee();
	double GetApogee();
	double GetCDrag();
	double GetBStar();
	double GetLaunchCycle();
	double GetAvoidanceSuccess();
	double GetExplosionProbability();
	double GetCollisionProbability();
	bool IsIntact();
	bool IsActive();
	bool IsPassive();
	CUDA_CALLABLE_MEMBER vector3D GetVelocity();
	vector3D GetPosition();
	vector<double> GetStateVector();
	CUDA_CALLABLE_MEMBER vector3D GetNormalVector();

	CUDA_CALLABLE_MEMBER OrbitalAnomalies GetAnomalies();
	CUDA_CALLABLE_MEMBER OrbitalElements& GetElements();

	bool SGP4Initialised();
	elsetrec& GetSGP4SatRec();

	void RemoveNFrag();
	void UpdateRAAN(double rightAscension);
	void UpdateArgP(double argPerigee);
	void RandomiseMeanAnomaly();
	void SetMeanAnomaly(double M);
	CUDA_CALLABLE_MEMBER void SetTrueAnomaly(double v);
	void SetEccentricAnomaly(double E);
	void UpdateOrbitalElements(vector3D deltaV);
	void UpdateOrbitalElements(OrbitalElements newElements);
	void UpdateOrbitalElements(vector3D position, vector3D velocity);
	void UpdateEpoch(double epochStep);
	void UpdateCollisionProbability(double probability);

	void SetID(long ID);
	void SetSourceID(long ID);
	void SetCentralBody(int c);
	void SetParentID(long ID);
	void SetCDrag(double cDrag);
	void SetBStar(double b);
	void SetInitEpoch(double epoch);
	void SetEpoch(double epoch);
	void SetRadius(double radii);
	void SetArea(double xsection);
	void SetMass(double newMass);
	void SetNFrag(int n);
	void SetConstellationID(int id);
	void SetLaunchCycle(double cycle);

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


class RemovedObject {
	//Object ID, Name, Object Type, Source Event, Mass, Remove Event, Remove Epoch, ParentID, SourceID, Source Type, Fragments Represented, State, Lifetime Collision Probability
public:
	long parentID, sourceID, objectID;
	double mass, removeEpoch, collisionProbability;
	bool isActive, isIntact, isPassive;
	string name;
	int sourceType, sourceEvent, removeEvent, objectType, nFrag;

	RemovedObject();
	RemovedObject(DebrisObject debris);
	long GetID();
	string GetName();
	int GetRemoveEvent();
	double GetRemoveEpoch();
	double GetMass();
	int	GetType();
	long GetSourceID();
	int GetSourceType();
	int GetSourceEvent();
	long GetParentID();
	bool IsIntact();
	bool IsActive();
	bool IsPassive();
	double GetCollisionProbability();
	int GetNFrag();


	void SetNewObjectID(long ID);
	void RemoveObject(int removeType, double epoch); // (0, 1, 2) = (Decay, Explosion, Collision) respectively.
};


bool CompareInitEpochs(DebrisObject objectA, DebrisObject objectB);
