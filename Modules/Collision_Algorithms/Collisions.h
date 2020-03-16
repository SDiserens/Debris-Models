#pragma once

//#include "stdafx.h"

#include "MOID\distlink.h"
#include "MOID\MOID.h"

class CollisionPair
{
public:
	DebrisObject primary, secondary;
	long primaryID, secondaryID;
	double approachAnomalyP, approachAnomalyS, probability;
	bool coplanar, collision;
	int overlapCount;
	Event tempEvent;
protected:
	double relativeInclination, relativeVelocity, deltaPrimary, deltaSecondary, deltaPrimary2, deltaSecondary2, boundingRadii, collisionAltitude;

public:
	CUDA_CALLABLE_MEMBER CollisionPair();
	CUDA_CALLABLE_MEMBER CollisionPair(DebrisObject& objectI, DebrisObject& objectJ);
	CUDA_CALLABLE_MEMBER CollisionPair(long IDI, long IDJ);
	CUDA_CALLABLE_MEMBER double GetRelativeInclination();
	vector<double>  CalculateAngularWindowPrimary(double distance);
	vector<double>  CalculateAngularWindowSecondary(double distance);
	vector3D GetPrimaryPositionAtTime(double timeFromEpoch);
	vector3D GetPrimaryVelocityAtTime(double timeFromEpoch);
	vector3D GetSecondaryPositionAtTime(double timeFromEpoch);
	vector3D GetSecondaryVelocityAtTime(double timeFromEpoch);
	double CalculateSeparationAtTime(double timeFromEpoch);
	CUDA_CALLABLE_MEMBER double CalculateMinimumSeparation();
	CUDA_CALLABLE_MEMBER double CalculateMinimumSeparation_DL();
	CUDA_CALLABLE_MEMBER double CalculateMinimumSeparation_MOID();
	CUDA_CALLABLE_MEMBER void CalculateArgumenstOfIntersection();
	CUDA_CALLABLE_MEMBER void CalculateArgumenstOfIntersectionCoplanar();
	CUDA_CALLABLE_MEMBER void CalculateRelativeInclination();
	CUDA_CALLABLE_MEMBER double GetBoundingRadii();
	CUDA_CALLABLE_MEMBER double GetCollisionAltitude();
	CUDA_CALLABLE_MEMBER void SetCollisionAltitude(double altitude);
	CUDA_CALLABLE_MEMBER void SetRelativeVelocity(double relV);
	CUDA_CALLABLE_MEMBER double GetRelativeVelocity();

protected:
	vector<double> CalculateAngularWindow(DebrisObject& object, double distance, double delta);
};

class CollisionAlgorithm
{
protected:
	bool relativeGravity = false, outputProbabilities;
	double elapsedTime, timeStep, pAThreshold;

protected:
	bool PerigeeApogeeTest(CollisionPair & objectPair);
	vector<double> collisionProbabilities;
	vector<Event> collisionList;
	vector<double> newCollisionProbabilities;
	vector<double> newCollisionAltitudes;
	vector<Event> newCollisionList;
	bool DetermineCollision(double collisionProbability);
	double CollisionCrossSection(DebrisObject& objectI, DebrisObject& objectJ);
	//double CalculateClosestApproach(CollisionPair objectPair);

	// Virtual function
	virtual double CollisionRate(CollisionPair &objectPair) = 0;
	virtual list<CollisionPair> CreatePairList(DebrisPopulation& population);
	virtual list<CollisionPair> CreatePairList_P(DebrisPopulation& population);

	vector<double> GetCollisionProbabilities();
	vector<double> GetNewCollisionProbabilities();

public:
	virtual void MainCollision(DebrisPopulation& population, double timeStep) = 0;
	virtual void MainCollision_P(DebrisPopulation& population, double timeStep);
	virtual void MainCollision_GPU(DebrisPopulation& population, double timeStep);
	virtual void SetThreshold(double threshold) = 0;
	virtual void SetMOID(int moid) = 0;

	void SwitchGravityComponent();
	vector<Event> GetCollisionList();
	vector<Event> GetNewCollisionList();
	double GetElapsedTime();

	vector<double> GetCollisionVerbose();
	vector<double> GetNewCollisionVerbose();
	vector<double> GetNewCollisionAltitudes();

	bool DetermineCollisionAvoidance(double avoidanceProbability);
};
