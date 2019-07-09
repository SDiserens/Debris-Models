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
		debris.second.UpdateOrbitalElements(UpdateElements(debris.second, timestep));
	}
}
