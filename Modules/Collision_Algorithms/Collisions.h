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
	vector<double> collisionProbabilities;
	vector<Event> collisionList;
	vector<double> newCollisionProbabilities;
	vector<double> newCollisionAltitudes;
	vector<Event> newCollisionList;
	double CollisionCrossSection(CollisionPair &objectPair);
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
	void SwitchParallelGPU();
	void SwitchParallelCPU();
	vector<Event> GetCollisionList();
	vector<Event> GetNewCollisionList();
	double GetElapsedTime();

	vector<double> GetCollisionVerbose();
	vector<double> GetNewCollisionVerbose();
	vector<double> GetNewCollisionAltitudes();

	bool DetermineCollision(double collisionProbability);
	bool DetermineCollisionAvoidance(double avoidanceProbability);
	bool UseGPU();
	bool UseParallel();
};
