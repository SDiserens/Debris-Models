#include "stdafx.h"
#include "DebrisPopulation.h"


DebrisPopulation::DebrisPopulation()
{
}


DebrisPopulation::~DebrisPopulation()
{
}

void DebrisPopulation::AddDebrisObject(DebrisObject debris)
{
	population.push_back(debris);
	populationCount++;
	totalMass += debris.GetMass();
}
