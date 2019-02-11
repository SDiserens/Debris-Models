#pragma once

#include "stdafx.h"

class CollisionAlgorithm
{
protected:
	bool relativeGravity = false, outputProbabilities;
	double elapsedTime;

protected:
	vector<double> collisionProbabilities;
	vector<pair<long, long>> collisionList;
	vector<double> newCollisionProbabilities;
	vector<pair<long, long>> newCollisionList;
	bool DetermineCollision(double collisionProbability);

	// Virtual function
	virtual double CollisionRate(DebrisObject& objectI, DebrisObject& objectJ) = 0;
	virtual vector<pair<long, long>> CreatePairList(DebrisPopulation& population) = 0;

public:
	void MainCollision(DebrisPopulation& population, double timeStep);
	void SwitchGravityComponent();
	vector<pair<long, long>> GetCollisionList();
	vector<double> GetCollisionProbabilities();
	vector<pair<long, long>> GetNewCollisionList();
	vector<double> GetNewCollisionProbabilities();

};