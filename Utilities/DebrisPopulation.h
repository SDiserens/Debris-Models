#pragma once
class Event
{

protected:
	static int eventSEQ;
	long eventID, debrisGenerated;
	int eventType; // ( 0 : Explosion, 1 : Collision,  2 : Collision Avoidance,)
	double eventEpoch, involvedMass;
	bool catastrophic, momentumConserved;

public:
	Event(double epoch, int type, bool consMomentum, bool catastrophic, double mass);
	Event(double epoch, int type, bool consMomentum, bool catastr, double mass, long debrisCount);
	~Event();
	string GetEventType();
};

class DebrisPopulation
{
protected:
	long populationCount;
	double totalMass, currentEpoch, startEpoch;

public:
	map<long, DebrisObject> population, removedPopulation;
	vector<Event> eventLog;

public:
	DebrisPopulation();
	~DebrisPopulation();
	double GetEpoch();
	int GetPopulationSize();
	void UpdateEpoch(double timeStep);
	void InitialiseEpoch(double epoch);
	void AddDebrisObject(DebrisObject debris);
	void AddDebrisEvent(Event debrisEvent);
	DebrisObject& GetObject(long ID);
	void DecayObject(long ID);
	void ExplodeObject(long ID);
	void CollideObject(long ID);
};

