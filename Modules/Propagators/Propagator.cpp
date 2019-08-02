#include "stdafx.h"
#include "Propagator.h"





Propagator::Propagator(DebrisPopulation & initPopulation) :population(initPopulation)
{
}


Propagator::~Propagator()
{
}

void Propagator::PropagatePopulation(double timestep)
{
	if (population.GetPopulationSize() > 0)
	{
		for (auto& debris : population.population)
		{
			UpdateElements(debris.second, timestep);
			if (population.GetPopulationSize() == 0)
				break;
		}
	}
	population.UpdateEpoch(timestep);
}
