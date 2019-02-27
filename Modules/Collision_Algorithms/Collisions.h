#pragma once

#include "stdafx.h"

class CollisionPair
{
public:
	DebrisObject primary, secondary;
	long primaryID, secondaryID;
	double approachAnomalyP, approachAnomalyS;
protected:
	double relativeInclination, deltaPrimary, deltaSecondary, boundingRadii;

public:
	CollisionPair(DebrisObject& objectI, DebrisObject& objectJ);
	CollisionPair(long IDI, long IDJ);
	double GetRelativeInclination();
	vector<pair<double, double>> CalculateAngularWindowPrimary(double distance);
	vector<pair<double, double>> CalculateAngularWindowSecondary(double distance);
	vector3D GetPrimaryPositionAtTime(double timeFromEpoch);
	vector3D GetPrimaryVelocityAtTime(double timeFromEpoch);
	vector3D GetSecondaryPositionAtTime(double timeFromEpoch);
	vector3D GetSecondaryVelocityAtTime(double timeFromEpoch);
	double CalculateSeparationAtTime(double timeFromEpoch);
	double CalculateMinimumSeparation();
	void CalculateArgumenstOfIntersection();
	void CalculateArgumenstOfIntersectionCoplanar();
	double GetBoundingRadii();

protected:
	void CalculateRelativeInclination();
	vector<pair<double, double>> CalculateAngularWindow(DebrisObject& object, double distance);
};

class CollisionAlgorithm
{
protected:
	bool relativeGravity = false, outputProbabilities;
	double elapsedTime, timeStep;

protected:
	vector<CollisionPair> CreatePairList(DebrisPopulation& population);
	vector<double> collisionProbabilities;
	vector<pair<long, long>> collisionList;
	vector<double> newCollisionProbabilities;
	vector<pair<long, long>> newCollisionList;
	bool DetermineCollision(double collisionProbability);
	double CalculateClosestApproach(CollisionPair objectPair);

	// Virtual function
	virtual double CollisionRate(DebrisObject& objectI, DebrisObject& objectJ) = 0;
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
