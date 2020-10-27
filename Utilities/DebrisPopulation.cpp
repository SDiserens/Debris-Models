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
	falseCollisionCount = 0;
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
	falseCollisionCount = 0;
}

double DebrisPopulation::GetNextInitEpoch()
{
	if (initEpochs.size() < 100)
		LoadLaunchTraffic(100 - initEpochs.size());
	if (initEpochs.size() == 0)
		return NAN;
	else
		return initEpochs.front().first;
}
double DebrisPopulation::GetTimeToNextInitEpoch()
{
	if (initEpochs.size() < 100)
		LoadLaunchTraffic(100 - initEpochs.size());
	if (initEpochs.size() == 0)
		return NAN;
	else
		return initEpochs.front().first - currentEpoch;
}
double DebrisPopulation::GetEpoch()
{
	return currentEpoch;
}

double DebrisPopulation::GetStartEpoch()
{
	return startEpoch;
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

void DebrisPopulation::SetLaunches(bool launch)
{
	launches = launch;
}

void DebrisPopulation::InitialiseEpoch(double epoch)
{
	currentEpoch = epoch;
	startEpoch = epoch;
}

void DebrisPopulation::LoadLaunchTraffic(int n)
{ 
	DebrisObject tempObject, newObject;
	int loadCount = min(n, (int) launchTraffic.size());

	for (int i=0; i < loadCount; i++) {
		tempObject = launchTraffic[0];
		AddDebrisObject(tempObject);
		launchTraffic.erase(launchTraffic.begin());

		newObject = CopyDebrisObject(tempObject);
		newObject.SetInitEpoch(newObject.GetInitEpoch() + newObject.GetLaunchCycle());
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
		populationCount += debris.GetNFrag();
		totalMass += debris.GetMass() * debris.GetNFrag();
		switch (debris.GetType()) {
		case 0: 
			upperStageCount += debris.GetNFrag();
			break;
		case 1: 
			spacecraftCount += debris.GetNFrag();
			break;
		case 2: 
			debrisCount += debris.GetNFrag();
			break;
		}
	}
	else if (debEpoch <= currentEpoch)
	{
		population.emplace(ID, debris);
		++populationCount;
		totalMass += debris.GetMass() * debris.GetNFrag();
		switch (debris.GetType()) {
		case 0:
			upperStageCount += debris.GetNFrag();
			break;
		case 1:
			spacecraftCount += debris.GetNFrag();
			break;
		case 2:
			debrisCount += debris.GetNFrag();
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

void DebrisPopulation::AddLaunchTraffic(vector<DebrisObject> launch)
{
	launchTraffic.insert(launchTraffic.end(), launch.begin(), launch.end());
	sort(launchTraffic.begin(), launchTraffic.end(), [](DebrisObject primary, DebrisObject secondary) {
		return (primary.GetInitEpoch() < secondary.GetInitEpoch());
	});
}

void DebrisPopulation::LoadPopulation()
{
	if (launches && (initEpochs.size() < 100))
		LoadLaunchTraffic(100 - initEpochs.size());
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
			populationCount += tempObject.GetNFrag();
			totalMass += tempObject.GetMass() * tempObject.GetNFrag();
			switch (tempObject.GetType()) {
			case 0:
				upperStageCount += tempObject.GetNFrag();
				break;
			case 1:
				spacecraftCount += tempObject.GetNFrag();
				break;
			case 2:
				debrisCount += tempObject.GetNFrag();
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
	++eventCount;
	if (type != 3) {
		debrisEvent.SetEventID();
		eventLog.push_back(debrisEvent);
		if (type == 0)
			++explosionCount;
		else if (type == 1)
			++collisionCount;
		else if (type == 2)
			++collisionAvoidanceCount;
	}
	else
		++falseCollisionCount;	
}

DebrisObject& DebrisPopulation::GetObject(long ID)
{
	if (population.count(ID) > 0)
		return population.at(ID);
	else if (loadingPopulation.count(ID) > 0)
		return loadingPopulation.at(ID);
	//else
		//return removedPopulation.at(ID);

}

bool DebrisPopulation::CheckObject(long ID)
{
	return (population.count(ID) > 0);
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
	RemovedObject tempObject;
	tempObject = RemovedObject(population[ID]);
	population.erase(ID);
	tempObject.RemoveObject(0, currentEpoch);
	removedPopulation.emplace(ID, tempObject);

	tempObject = RemovedObject(removedPopulation[ID]);
	int nFrag = tempObject.GetNFrag();
	totalMass -= tempObject.GetMass() * nFrag;
	populationCount -= nFrag;
	switch (tempObject.GetType()) {
		case 0: 
			upperStageCount -= nFrag;
			break;
		case 1: 
			spacecraftCount -= nFrag;
			break;
		case 2: 
			debrisCount -= nFrag;
			break;
	}

}

void DebrisPopulation::ExplodeObject(long ID)
{
	DebrisObject& tempObject(population[ID]);
	RemovedObject newObject(tempObject);
	newObject.RemoveObject(1, currentEpoch);
	if (tempObject.GetNFrag() > 1)
	{
		tempObject.RemoveNFrag();
		newObject.nFrag = 1;
		newObject.SetNewObjectID();
	}
	else {
		population.erase(ID);
		newObject.RemoveObject(1, currentEpoch);
	}
	removedPopulation.emplace(newObject.GetID(), newObject);
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
	DebrisObject& tempObject(population[ID]);
	RemovedObject newObject(tempObject);
	newObject.RemoveObject(2, currentEpoch);
	if (tempObject.GetNFrag() > 1)
	{
		tempObject.RemoveNFrag();
		newObject.nFrag = 1;
		newObject.SetNewObjectID();
	}
	else {
		population.erase(ID);
		newObject.RemoveObject(1, currentEpoch);
	}
	removedPopulation.emplace(newObject.GetID(), newObject);
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

