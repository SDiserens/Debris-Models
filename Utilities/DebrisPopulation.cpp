#include "stdafx.h"
#include "DebrisPopulation.h"


int Event::eventSEQ = 0;

DebrisPopulation::DebrisPopulation()
{
	currentEpoch = 0;
	startEpoch = 0;
	populationCount = 0;
	totalMass = 0;
}


DebrisPopulation::~DebrisPopulation()
{
}

double DebrisPopulation::GetNextInitEpoch()
{
	if (initEpochs.size() == 0)
		return NAN;
	else
		return initEpochs[0].first;
}

double DebrisPopulation::GetEpoch()
{
	return currentEpoch;
}

int DebrisPopulation::GetPopulationSize()
{
	return populationCount;
}

int DebrisPopulation::GetScalingPower()
{
	return scalingPower;
}

double DebrisPopulation::GetAverageSMA()
{
	return averageSemiMajorAxis;
}

void DebrisPopulation::SetAverageSMA(double averageSMA)
{
	averageSemiMajorAxis = averageSMA;
}

void DebrisPopulation::SetScalingPower(int power)
{
	scalingPower = power;
}

void DebrisPopulation::InitialiseEpoch(double epoch)
{
	currentEpoch = epoch;
	startEpoch = epoch;
}

void DebrisPopulation::UpdateEpoch(double timeStep)
{
	currentEpoch += timeStep;
}

void DebrisPopulation::AddDebrisObject(DebrisObject debris)
{
	double debEpoch = debris.GetInitEpoch();
	long ID = debris.GetID();


	if (isnan(debEpoch))
	{
		debris.SetInitEpoch(currentEpoch);
		population.emplace(ID, debris);
		populationCount++;
		totalMass += debris.GetMass();
	}
	else if (debEpoch <= currentEpoch)
	{
		population.emplace(ID, debris);
		populationCount++;
		totalMass += debris.GetMass();
	}
	else
	{
		loadingPopulation.emplace(ID, debris);
		initEpochs.push_back(make_pair(debEpoch, ID));
		sort(initEpochs.begin(), initEpochs.end());
	}
}


void DebrisPopulation::LoadPopulation()
{
	for (auto ID : initEpochs)
	{
		if (ID.first <= currentEpoch)
		{
			DebrisObject tempObject(loadingPopulation[ID.second]);
			loadingPopulation.erase(ID.second);
			population.emplace(ID.second, tempObject);
			populationCount++;
			totalMass += tempObject.GetMass();
		}
	}

}

void DebrisPopulation::AddDebrisEvent(Event debrisEvent)
{
	eventLog.push_back(debrisEvent);
}

DebrisObject& DebrisPopulation::GetObject(long ID)
{
	if (population.count(ID) > 0)
		return population.at(ID);
	else if (loadingPopulation.count(ID) > 0)
		return loadingPopulation.at(ID);
	else
		return removedPopulation.at(ID);

}

void DebrisPopulation::DecayObject(long ID)
{
	DebrisObject tempObject(population[ID]);
	population.erase(ID);
	tempObject.RemoveObject(0, currentEpoch);
	removedPopulation.emplace(ID, tempObject);
	populationCount--;
}

void DebrisPopulation::ExplodeObject(long ID)
{
	DebrisObject tempObject(population[ID]);
	population.erase(ID);
	tempObject.RemoveObject(1, currentEpoch);
	removedPopulation.emplace(ID, tempObject);
	populationCount--;
}

void DebrisPopulation::CollideObject(long ID)
{
	DebrisObject tempObject(population[ID]);
	population.erase(ID);
	tempObject.RemoveObject(2, currentEpoch);
	removedPopulation.emplace(ID, tempObject);
	populationCount--;
}

Event::Event(double epoch, int type, bool consMomentum, bool catastr, double mass)
{
	eventID = ++eventSEQ;
	eventEpoch = epoch;
	eventType = type;
	momentumConserved = consMomentum;
	catastrophic = catastr;
	involvedMass = mass;
}

Event::Event(double epoch, int type, bool consMomentum, bool catastr, double mass, long debrisCount)
{
	eventID = ++eventSEQ;
	eventEpoch = epoch;
	eventType = type;
	momentumConserved = consMomentum;
	catastrophic = catastr;
	debrisGenerated = debrisCount;
	involvedMass = mass;
}

Event::~Event()
{
}

string Event::GetEventType()
{
	string eventName;
	if (eventType == 0)
		eventName = "Explosion";
	else if (eventType == 1)
		if (catastrophic)
			eventName = "Catastrophic_Collision";
		else
			eventName = "NonCatastrophic_Collision";
	else if (eventType == 2)
		eventName = "Collision Avoidance";
	return eventName;
}
