#pragma once

#include "stdafx.h"
#include "Collisions.h"
#include <thrust\device_vector.h>

class OrbitTrace : public CollisionAlgorithm
{
	//double deltaR, deltaB;
	int MOIDtype = 0; // {0: inbuilt Newton; 1: distlink; 2: MOID}

public:
	void MainCollision(DebrisPopulation& population, double timestep);
	void MainCollision_P(DebrisPopulation& population, double timestep);
	void MainCollision_GPU(DebrisPopulation& population, double timestep);
	//void MainCollision_GPU_Cuda(DebrisPopulation& population, double timestep);
	void SetThreshold(double threshold);
	void SetMOID(int moid);
	OrbitTrace(bool probabilities = false, double threshold = 10);
	OrbitTrace(bool probabilities = false, double threshold = 10, int moid = 0);

protected:
	double CollisionRate(CollisionPair &objectPair);
	thrust::host_vector<CollisionPair> CreatePairList_GPU(DebrisPopulation & population);
	/*
	double CalculateSpatialDensity(DebrisObject object, double radius, double latitude);
	double CalculateRadialSpatialDensity(DebrisObject object, double radius);
	double CalculateLatitudinalSpatialDensityRatio(DebrisObject object, double latitude);
	double CalculateVolumeElement(double radius, double latitude);
	*/
	bool CoplanarFilter(CollisionPair objectPair);
	bool HeadOnFilter(CollisionPair objectPair);
	bool SynchronizedFilter(CollisionPair objectPair);
	bool ProximityFilter(CollisionPair objectPair);

};

