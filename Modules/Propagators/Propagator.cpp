#include "stdafx.h"
#include "Propagator.h"


Propagator::Propagator(DebrisPopulation & initPopulation)
{
	population = initPopulation
}


Propagator::~Propagator()
{
}

void Propagator::PropagatePopulation(double timestep)
{
	for (auto& debris : population.population)
	{
		UpdateElements(debris.second, timestep);
	}
	population.UpdateEpoch(timestep);
}
