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

void DebrisPopulation::Clear()
{
	population.clear();
	removedPopulation.clear();
	loadingPopulation.clear();
	initEpochs.clear();
	eventLog.clear();

	currentEpoch = 0;
	startEpoch = 0;
	populationCount = 0;
	totalMass = 0;

}

double DebrisPopulation::GetNextInitEpoch()
{
	if (initEpochs.size() == 0)
		return NAN;
	else
		return initEpochs.front().first;
}
double DebrisPopulation::GetTimeToNextInitEpoch()
{
	if (initEpochs.size() == 0)
		return NAN;
	else
		return initEpochs.front().first - currentEpoch;
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

void DebrisPopulation::SetDuration(double duration)
{
	durationDays = duration;
}

double DebrisPopulation::GetDuration()
{
	return durationDays;
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
	if (initEpochs.size() != 0)
	{
		pair<double, long> ID;
		while (initEpochs.front().first <= currentEpoch)
		{
			ID = initEpochs.front();
			initEpochs.erase(initEpochs.begin());
			DebrisObject tempObject(loadingPopulation[ID.second]);
			loadingPopulation.erase(ID.second);
			population.emplace(ID.second, tempObject);
			populationCount++;
			totalMass += tempObject.GetMass();

			if (initEpochs.size() == 0)
				break;
		}
	}
	

}

void DebrisPopulation::AddDebrisEvent(Event debrisEvent)
{
	int type = debrisEvent.GetEventType();
	eventLog.push_back(debrisEvent);
	eventCount++;
	if (type == 0)
		explosionCount++;
	else if (type == 1)
		collisionCount++;
	else if (type == 2)
		collisionAvoidanceCount++;
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

int DebrisPopulation::GetEventCount()
{
	return eventCount;
}

int DebrisPopulation::GetExplosionCount()
{
	return explosionCount;
}

int DebrisPopulation::GetCollsionCount()
{
	return collisionCount;
}

int DebrisPopulation::GetCAMCount()
{
	return collisionAvoidanceCount;
}

tuple<double, int, int, tuple<int, int, int>> DebrisPopulation::GetPopulationState()
{
	return make_tuple(currentEpoch, populationCount, eventCount, make_tuple(explosionCount, collisionCount, collisionAvoidanceCount));
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

int Event::GetEventType()
{
	return eventType;
}

string Event::GetEventTypeString()
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
