#pragma once

#include "stdafx.h"

class CollisionAlgorithm
{
protected:
	vector<double> collisionProbabilities;
	vector<pair<long, long>> collisionList;

public:
	vector<pair<long, long>> GetCollisionList();
	vector<double> GetCollisionProbabilities();

};