#pragma once

#include "stdafx.h"
#include "Collisions.h"

class OrbitTrace : public CollisionAlgorithm
{
	//double deltaR, deltaB;

public:
	void MainCollision(DebrisPopulation& population, double timeStep);
	void MainCollision_P(DebrisPopulation& population, double timeStep);
	void SetThreshold(double threshold);
	OrbitTrace(bool probabilities = false, double threshold = 10);

protected:
	double CollisionRate(CollisionPair &objectPair);

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

