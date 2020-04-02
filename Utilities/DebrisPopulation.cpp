#include "stdafx.h"


DebrisPopulation::DebrisPopulation()
{
	currentEpoch = 0;
	startEpoch = 0;
	populationCount = 0;
	totalMass = 0;
	upperStageCount = 0;
	spacecraftCount = 0;
	debrisCount = 0;
	eventCount = 0;
	explosionCount = 0; 
	collisionCount = 0;
	collisionAvoidanceCount = 0;
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

	upperStageCount = 0;
	spacecraftCount = 0;
	debrisCount = 0;
	eventCount = 0;
	explosionCount = 0;
	collisionCount = 0;
	collisionAvoidanceCount = 0;
}

double DebrisPopulation::GetNextInitEpoch()
{
	if (initEpochs.size() == 0)
		LoadLaunchTraffic();
	if (initEpochs.size() == 0)
		return NAN;
	else
		return initEpochs.front().first;
}
double DebrisPopulation::GetTimeToNextInitEpoch()
{
	if (initEpochs.size() == 0)
		LoadLaunchTraffic();
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

void DebrisPopulation::LoadLaunchTraffic()
{ 
	DebrisObject tempObject, newObject;
	int loadCount = min(100, (int) launchTraffic.size());

	for (int i=0; i < loadCount; i++) {
		tempObject = launchTraffic[0];
		AddDebrisObject(tempObject);
		launchTraffic.erase(launchTraffic.begin());

		newObject = CopyDebrisObject(tempObject);
		newObject.SetInitEpoch(newObject.GetInitEpoch() + launchCycle);
		launchTraffic.push_back(newObject);
	}
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
		++populationCount;
		totalMass += debris.GetMass();
		switch (debris.GetType()) {
		case 0: 
			++upperStageCount;
			break;
		case 1: 
			++spacecraftCount;
			break;
		case 2: 
			++debrisCount;
			break;
		}
	}
	else if (debEpoch <= currentEpoch)
	{
		population.emplace(ID, debris);
		++populationCount;
		totalMass += debris.GetMass();
		switch (debris.GetType()) {
		case 0: 
			++upperStageCount;
			break;
		case 1: 
			++spacecraftCount;
			break;
		case 2: 
			++debrisCount;
			break;
		}
	}
	else
	{
		loadingPopulation.emplace(ID, debris);
		initEpochs.push_back(make_pair(debEpoch, ID));
		sort(initEpochs.begin(), initEpochs.end());
	}
}

void DebrisPopulation::AddDefinedEvent(Event breakup)
{
	if (definedEvents.size() > 0) {
		for (int n = 0; n < definedEvents.size(); n++) {
			if (breakup.GetEventEpoch() < definedEvents[n].GetEventEpoch()) {
				definedEvents.emplace(definedEvents.begin() + n, breakup);
				break;
			}
		}
	}
	else
		definedEvents.push_back(breakup);
}

void DebrisPopulation::AddLaunchTraffic(vector<DebrisObject> launch, double cycle)
{
	launchCycle = cycle;
	launchTraffic = launch;
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
			++populationCount;
			totalMass += tempObject.GetMass();
			switch (tempObject.GetType()) {
			case 0: 
				++upperStageCount;
				break;
			case 1: 
				++spacecraftCount;
				break;
			case 2: 
				++debrisCount;
				break;
			}
			if (initEpochs.size() == 0)
				break;
		}
	}
	

}

void DebrisPopulation::AddDebrisEvent(Event debrisEvent)
{
	int type = debrisEvent.GetEventType();
	debrisEvent.SetEventID();
	eventLog.push_back(debrisEvent);
	++eventCount;
	if (type == 0)
		++explosionCount;
	else if (type == 1)
		++collisionCount;
	else if (type == 2)
		++collisionAvoidanceCount;
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

vector<Event> DebrisPopulation::GenerateExplosionList()
{
	vector<Event> explosionList;
	for (auto&  object : population) {
		if (object.second.IsIntact() && object.second.GetType() != 2) {
			if (object.second.GetExplosionProbability() > randomNumber()) {
				Event tempExplosion(currentEpoch, object.first, object.second.GetMass());
				explosionList.push_back(tempExplosion);
			}
		}
	}
	return explosionList;
}

vector<Event> DebrisPopulation::GenerateDefinedEventList()
{
	return definedEvents;
}

void DebrisPopulation::DecayObject(long ID)
{
	DebrisObject tempObject(population[ID]);
	population.erase(ID);
	tempObject.RemoveObject(0, currentEpoch);
	removedPopulation.emplace(ID, tempObject);
	totalMass -= tempObject.GetMass();
	populationCount--;
	switch (tempObject.GetType()) {
		case 0: 
			upperStageCount--;
			break;
		case 1: 
			spacecraftCount--;
			break;
		case 2: 
			debrisCount--;
			break;
	}
}

void DebrisPopulation::ExplodeObject(long ID)
{
	DebrisObject tempObject(population[ID]);
	population.erase(ID);
	tempObject.RemoveObject(1, currentEpoch);
	removedPopulation.emplace(ID, tempObject);
	totalMass -= tempObject.GetMass();
	populationCount--;
	switch (tempObject.GetType()) {
	case 0:
		upperStageCount--;
		break;
	case 1:
		spacecraftCount--;
		break;
	case 2:
		debrisCount--;
		break;
	}
}

void DebrisPopulation::CollideObject(long ID)
{
	DebrisObject tempObject(population[ID]);
	population.erase(ID);
	tempObject.RemoveObject(2, currentEpoch);
	removedPopulation.emplace(ID, tempObject);
	totalMass -= tempObject.GetMass();
	populationCount--;
	switch (tempObject.GetType()) {
	case 0:
		upperStageCount--;
		break;
	case 1:
		spacecraftCount--;
		break;
	case 2:
		debrisCount--;
		break;
	}
}

void DebrisPopulation::RemoveObject(long ID, int type) // type: (0 = explosion; 1 = collision; 2 = decay)
{
	if (type == 0)
		ExplodeObject(ID);
	else if (type == 1)
		CollideObject(ID);
	else if (type == 2)
		DecayObject(ID);
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

tuple<double, int, tuple<int, int, int>, int, tuple<int, int, int>> DebrisPopulation::GetPopulationState()
{
	return make_tuple(currentEpoch, populationCount, make_tuple(upperStageCount, spacecraftCount, debrisCount) ,eventCount, make_tuple(explosionCount, collisionCount, collisionAvoidanceCount));
}

vector<Event> DebrisPopulation::GetEventLog()
{
	return eventLog;
}

