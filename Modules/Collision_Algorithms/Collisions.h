#pragma once

#include "stdafx.h"

class CollisionPair
{
public:
	DebrisObject primary, secondary;
	long primaryID, secondaryID;
	double approachAnomalyP, approachAnomalyS;
	bool coplanar;
protected:
	double relativeInclination, deltaPrimary, deltaSecondary, deltaPrimary2, deltaSecondary2, boundingRadii;

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

protected:
	vector<double> CalculateAngularWindow(DebrisObject& object, double distance, double delta);
};

class CollisionAlgorithm
{
protected:
	bool relativeGravity = false, outputProbabilities;
	double elapsedTime, timeStep, pAThreshold;

protected:
	vector<CollisionPair> CreatePairList(DebrisPopulation& population);
	bool PerigeeApogeeTest(CollisionPair & objectPair);
	vector<double> collisionProbabilities;
	vector<pair<long, long>> collisionList;
	vector<double> newCollisionProbabilities;
	vector<pair<long, long>> newCollisionList;
	bool DetermineCollision(double collisionProbability);
	//double CalculateClosestApproach(CollisionPair objectPair);

	// Virtual function
	virtual double CollisionRate(CollisionPair &objectPair) = 0;
	//virtual vector<pair<long, long>> CreatePairList(DebrisPopulation& population) = 0;
	double CollisionCrossSection(DebrisObject& objectI, DebrisObject& objectJ);


public:
	void SwitchGravityComponent();
	vector<pair<long, long>> GetCollisionList();
	vector<double> GetCollisionProbabilities();
	vector<pair<long, long>> GetNewCollisionList();
	vector<double> GetNewCollisionProbabilities();
	double GetElapsedTime();

};
