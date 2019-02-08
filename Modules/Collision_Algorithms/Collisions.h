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
	double CollisionRate(DebrisObject& objectI, DebrisObject& objectJ) { cerr << "Error. CollisionRate not available for base type." << endl; };;
	vector<pair<long, long>> CreatePairList(DebrisPopulation& population){ cerr << "Error. CreatePairList not available for base type." << endl; };

public:
	void MainCollision(DebrisPopulation& population, double timeStep);
	void SwitchGravityComponent();
	vector<pair<long, long>> GetCollisionList();
	vector<double> GetCollisionProbabilities();
	vector<pair<long, long>> GetNewCollisionList();
	vector<double> GetNewCollisionProbabilities();

};