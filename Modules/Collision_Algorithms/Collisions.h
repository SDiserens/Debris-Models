#pragma once

//#include "stdafx.h"

class CollisionPair
{
public:
	DebrisObject primary, secondary;
	long primaryID, secondaryID;
	double approachAnomalyP, approachAnomalyS;
	bool coplanar;
	int overlapCount;
protected:
	double relativeInclination, relativeVelocity, deltaPrimary, deltaSecondary, deltaPrimary2, deltaSecondary2, boundingRadii, collisionAltitude;

public:
	CollisionPair(DebrisObject& objectI, DebrisObject& objectJ);
	CollisionPair(long IDI, long IDJ);
	double GetRelativeInclination();
	vector<double>  CalculateAngularWindowPrimary(double distance);
	vector<double>  CalculateAngularWindowSecondary(double distance);
	vector3D GetPrimaryPositionAtTime(double timeFromEpoch);
	vector3D GetPrimaryVelocityAtTime(double timeFromEpoch);
	vector3D GetSecondaryPositionAtTime(double timeFromEpoch);
	vector3D GetSecondaryVelocityAtTime(double timeFromEpoch);
	double CalculateSeparationAtTime(double timeFromEpoch);
	double CalculateMinimumSeparation();
	void CalculateArgumenstOfIntersection();
	void CalculateArgumenstOfIntersectionCoplanar();
	void CalculateRelativeInclination();
	double GetBoundingRadii();
	double GetCollisionAltitude();
	void SetCollisionAltitude(double altitude);
	void SetRelativeVelocity(double relV);
	double GetRelativeVelocity();

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
	virtual void SetThreshold(double threshold) = 0;
	void SwitchGravityComponent();
	vector<Event> GetCollisionList();
	vector<Event> GetNewCollisionList();
	double GetElapsedTime();

	vector<double> GetCollisionVerbose();
	vector<double> GetNewCollisionVerbose();
	vector<double> GetNewCollisionAltitudes();

	bool DetermineCollisionAvoidance(double avoidanceProbability);
};
