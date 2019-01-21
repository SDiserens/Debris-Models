#include "stdafx.h"
#include "DebrisPopulation.h"


int Event::eventSEQ = 0;

DebrisPopulation::DebrisPopulation()
{
	currentEpoch = startEpoch = 0;
}


DebrisPopulation::~DebrisPopulation()
{
}

double DebrisPopulation::GetEpoch()
{
	return currentEpoch;
}

void DebrisPopulation::InitialiseEpoch(double epoch)
{
	currentEpoch = startEpoch = epoch;
}

void DebrisPopulation::UpdateEpoch(double timeStep)
{
	currentEpoch += timeStep;
}

void DebrisPopulation::AddDebrisObject(DebrisObject debris)
{
	population.emplace(debris.GetID(), debris);
	populationCount++;
	totalMass += debris.GetMass();
}


void DebrisPopulation::AddDebrisEvent(Event debrisEvent)
{
	eventLog.push_back(debrisEvent);
}

DebrisObject DebrisPopulation::GetObject(long ID)
{
	return population[ID];
}

void DebrisPopulation::DecayObject(long ID)
{
	DebrisObject tempObject(population[ID]);
	population.erase(ID);
	tempObject.RemoveObject(0, currentEpoch);
	removedPopulation.emplace(ID, tempObject);
}

void DebrisPopulation::ExplodeObject(long ID)
{
	DebrisObject tempObject(population[ID]);
	population.erase(ID);
	tempObject.RemoveObject(1, currentEpoch);
	removedPopulation.emplace(ID, tempObject);
}

void DebrisPopulation::CollideObject(long ID)
{
	DebrisObject tempObject(population[ID]);
	population.erase(ID);
	tempObject.RemoveObject(2, currentEpoch);
	removedPopulation.emplace(ID, tempObject);
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
