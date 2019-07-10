#include "stdafx.h"
#include "Propagator.h"


Propagator::Propagator()
{
}


Propagator::~Propagator()
{
}

void Propagator::PropagatePopulation(DebrisPopulation & population, double timestep)
{
	for (auto& debris : population.population)
	{
		UpdateElements(debris.second, timestep);
	}
}
