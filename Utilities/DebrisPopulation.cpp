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
	int type = debris.GetType();

	if (type == 0)
		upperStageCount++;
	else if (type == 1)
		spacecraftCount++;
	else if (type == 2)
		debrisCount++;

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

vector<Event> DebrisPopulation::GenerateExplosionList()
{
	vector<Event> explosionList;
	for (auto&  object : population) {
		//ToDo introduce explosion pre-filtering based on isIntact
		if (object.second.GetType() != 2) {
			if (object.second.GetExplosionProbability() > randomNumber()) {
				Event tempExplosion(currentEpoch, object.first, object.second.GetMass());
				explosionList.push_back(tempExplosion);
			}
		}
	}
	return explosionList;
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

Event::Event()
{
}

Event::Event(double epoch, long objectID, double mass)
{
	eventID = ++eventSEQ;
	eventEpoch = epoch;
	eventType = 0;
	involvedMass = mass;
	primaryID = objectID;
	secondaryID = -1;
}

Event::Event(double epoch, long objectID, bool consMomentum, bool catastr, double mass, long debrisCount)
{
	eventID = ++eventSEQ;
	eventEpoch = epoch;
	eventType = 0;
	momentumConserved = consMomentum;
	catastrophic = catastr;
	debrisGenerated = debrisCount;
	involvedMass = mass;
	primaryID = objectID;
	secondaryID = -1;
}



Event::Event(double epoch, long targetID, long projectileID, double relV, double mass, double alt)
{
	eventID = ++eventSEQ;
	eventEpoch = epoch;
	eventType = 1;
	involvedMass = mass;
	relativeVelocity = relV;
	primaryID = targetID;
	secondaryID = projectileID;
	altitude = alt;
}

Event::~Event()
{
}

void Event::CollisionAvoidance()
{
	eventType = 2;
}

void Event::SwapPrimarySecondary()
{
	long tempID = primaryID;
	primaryID = secondaryID;
	secondaryID = tempID;
}

void Event::SetConservationMomentum(bool conservedFlag)
{
	momentumConserved = conservedFlag;
}


void Event::SetCatastrophic(bool catastrophicFlag)
{
	catastrophic = catastrophicFlag;
}

void Event::SetEMR(double eMRatio)
{
	energyMassRatio = eMRatio;
}

void Event::SetDebrisCount(long count)
{
	debrisGenerated = count;
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

double Event::GetEventEpoch()
{
	return eventEpoch;
}

long Event::GetPrimary()
{
	return primaryID;
}

long Event::GetSecondary()
{
	return secondaryID;
}

pair<long, long> Event::GetCollisionPair()
{
	return make_pair(primaryID, secondaryID);
}
