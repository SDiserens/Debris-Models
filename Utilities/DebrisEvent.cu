#include "stdafx.h"

int Event::eventSEQ = 0;

Event::Event()
{
}

Event::Event(double epoch, long objectID, double mass)
{
	eventEpoch = epoch;
	eventType = 0;
	involvedMass = mass;
	primaryID = objectID;
	secondaryID = -1;
}

Event::Event(double epoch, long objectID, bool consMomentum, bool catastr, double mass, long debrisCount)
{
	eventEpoch = epoch;
	eventType = 0;
	momentumConserved = consMomentum;
	catastrophic = catastr;
	debrisGenerated = debrisCount;
	involvedMass = mass;
	primaryID = objectID;
	secondaryID = -1;

}



Event::Event(double epoch, long targetID, long projectileID, double relV, double mass, double alt, double separation, double probability)
{
	eventEpoch = epoch;
	eventType = 1;
	involvedMass = mass;
	relativeVelocity = relV;
	primaryID = targetID;
	secondaryID = projectileID;
	altitude = alt;
	minSeparation = separation;
	collisionProbability = probability;
}


void Event::SetAltitude(double alt) {
	altitude = alt;
}

void Event::SetCollisionAnomalies(double primaryV, double secondaryV)
{
	primaryAnomaly = primaryV;
	secondaryAnomaly = secondaryV;
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

void Event::SetEpoch(double epoch)
{
	eventEpoch = epoch;
}

void Event::SetEventID()
{
	eventID = ++eventSEQ;
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