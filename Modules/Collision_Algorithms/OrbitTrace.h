#pragma once

#include "stdafx.h"
#include "Collisions.h"

class OrbitTrace : public CollisionAlgorithm
{
	double deltaR, deltaB;

public:
	void MainCollision(DebrisPopulation& population, double timeStep);

protected:
	double CollisionRate(DebrisObject& objectI, DebrisObject& objectJ);
	vector<pair<long, long>> CreatePairList(DebrisPopulation& population);
	double CalculateSpatialDensity(DebrisObject object, double radius, double latitude);
	double CalculateRadialSpatialDensity(DebrisObject object, double radius);
	double CalculateLatitudinalSpatialDensityRatio(DebrisObject object, double latitude);
	double CalculateVolumeElement(double radius, double latitude);
};