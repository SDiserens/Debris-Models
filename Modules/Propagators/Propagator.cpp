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
	population.UpdateEpoch(timestep);
	SyncPopulation();
}

void Propagator::SyncPopulation()
{
	double currEpoch, debrisEpoch, timestep;

	if (population.GetPopulationSize() > 0)
	{
		currEpoch = population.GetEpoch();
		for (auto& debris : population.population)
		{
			debrisEpoch = debris.second.GetEpoch();
			if (debrisEpoch < currEpoch) {
				timestep = currEpoch - debrisEpoch;
				UpdateElements(debris.second, timestep);
				debris.second.SetEpoch(currEpoch);
			}
			if (population.GetPopulationSize() == 0)
				break;
		}
	}
}

double Propagator::CalculateBStar(DebrisObject & object)
{
	double ballisticC;

	ballisticC = object.GetCDrag() * object.GetAreaToMass();

	return rhoZero / 2 * ballisticC;
}
