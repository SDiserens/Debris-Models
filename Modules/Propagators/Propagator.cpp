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
	population.LoadPopulation();

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

double Propagator::CalculateBStar(DebrisObject & object)
{
	double ballisticC;

	ballisticC = object.GetCDrag() * object.GetAreaToMass();

	return rhoZero / 2 * ballisticC;
}
