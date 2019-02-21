#pragma once

#include "stdafx.h"

class CollisionAlgorithm
{
protected:
	bool relativeGravity = false, outputProbabilities;
	double elapsedTime;

protected:
	vector<CollisionPair> CreatePairList(DebrisPopulation& population);
	vector<double> collisionProbabilities;
	vector<pair<long, long>> collisionList;
	vector<double> newCollisionProbabilities;
	vector<pair<long, long>> newCollisionList;
	bool DetermineCollision(double collisionProbability);

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

class CollisionPair
{
public:
	DebrisObject primary, secondary;
protected:
	double relativeInclination, deltaPrimary, deltaSecondary;

public:
	CollisionPair(DebrisObject& objectI, DebrisObject& objectJ);
	double GetRelativeInclination();
	vector<pair<double, double>> CalculateAngularWindowPrimary(double distance);
	vector<pair<double, double>> CalculateAngularWindowSecondary(double distance);
	vector3D GetPrimaryPositionAtTime(double timeFromEpoch);
	vector3D GetPrimaryVelocityAtTime(double timeFromEpoch);
	vector3D GetSecondaryPositionAtTime(double timeFromEpoch);
	vector3D GetSecondaryVelocityAtTime(double timeFromEpoch);
	double CalculateSeparationAtTime(double timeFromEpoch);
	double CalculateMinimumSeparation();

protected:
	void CalculateRelativeInclination();
	void CalculateArgumenstOfIntersection();
	vector<pair<double, double>> CalculateAngularWindow(DebrisObject& object, double distance);
};