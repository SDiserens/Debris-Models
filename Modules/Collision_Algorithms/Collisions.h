#pragma once

#include "stdafx.h"

class CollisionAlgorithm
{
protected:
	vector<double> collisionProbabilities;
	vector<pair<long, long>> collisionList;
	vector<double> newCollisionProbabilities;
	vector<pair<long, long>> newCollisionList;

public:
	vector<pair<long, long>> GetCollisionList();
	vector<double> GetCollisionProbabilities();
	vector<pair<long, long>> GetNewCollisionList();
	vector<double> GetNewCollisionProbabilities();
};