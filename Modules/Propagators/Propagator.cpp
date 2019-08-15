#include "stdafx.h"
#include "Propagator.h"


Propagator::Propagator(DebrisPopulation & initPopulation) : population(initPopulation)
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

	if (population.GetPopulationSize() > 0)
	{
		double currEpoch, debrisEpoch, timestep;

		currEpoch = population.GetEpoch();
		for (auto& debris : population.population)
		{
			debrisEpoch = debris.second.GetEpoch();
			if (debrisEpoch <= currEpoch) {
				timestep = currEpoch - debrisEpoch;
				UpdateElements(debris.second, secondsDay * timestep);
				debris.second.SetEpoch(currEpoch);
			}
			if (population.GetPopulationSize() == 0)
				break;
		}
		RemovePopulation();
	}
}

void Propagator::RemovePopulation()
{
	for (long ID : removeID) {

		population.DecayObject(ID);
	}
	removeID.clear();
}

double Propagator::CalculateBStar(DebrisObject & object)
{
	double ballisticC;

	ballisticC = object.GetCDrag() * object.GetAreaToMass();

	return rhoZero / 2 * ballisticC;
}
