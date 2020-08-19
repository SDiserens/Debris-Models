#pragma once

//#include "stdafx.h"

#include "CollisionPair.cuh"

class CollisionAlgorithm
{
	bool GPU = false, parallel = false;
protected:
	bool relativeGravity = false, outputProbabilities;
	double elapsedTime, timeStep, pAThreshold;

protected:
	bool PerigeeApogeeTest(CollisionPair & objectPair);
	vector<Event> newCollisionList;
	double CollisionCrossSection(CollisionPair &objectPair);
	//double CalculateClosestApproach(CollisionPair objectPair);

	// Virtual function
	virtual double CollisionRate(CollisionPair &objectPair) = 0;
	virtual list<CollisionPair> CreatePairList(DebrisPopulation& population);
	virtual list<CollisionPair> CreatePairList_P(DebrisPopulation& population);

public:
	virtual void MainCollision(DebrisPopulation& population, double timeStep) = 0;
	virtual void MainCollision_P(DebrisPopulation& population, double timeStep);
	virtual void MainCollision_GPU(DebrisPopulation& population, double timeStep);
	virtual void SetThreshold(double threshold) = 0;
	virtual void SetMOID(int moid) = 0;

	void SwitchGravityComponent();
	void SwitchParallelGPU();
	void SwitchParallelCPU();
	vector<Event> GetNewCollisionList();
	double GetElapsedTime();
	
	bool DetermineCollision(double collisionProbability);
	bool DetermineCollisionAvoidance(double avoidanceProbability);
	bool CheckValidCollision(DebrisObject target, DebrisObject projectile);
	bool UseGPU();
	bool UseParallel();
};
