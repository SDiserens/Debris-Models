#pragma once
class Event
{

public:
	static int eventSEQ;
	long eventID, debrisGenerated, primaryID, secondaryID;
	int eventType; // ( 0 : Explosion, 1 : Collision,  2 : Collision Avoidance,)
	double eventEpoch, altitude, involvedMass, relativeVelocity;
	bool catastrophic, momentumConserved;

public:
	Event();
	Event(double epoch, long objectID, bool consMomentum, bool catastrophic, double mass);
	Event(double epoch, long objectID, bool consMomentum, bool catastr, double mass, long debrisCount);
	Event(double epoch, long targetID, long projectileID, double relV, bool consMomentum, bool catastr, double mass);
	Event(double epoch, long targetID, long projectileID, double relV, bool consMomentum, bool catastr, double mass, long debrisCount);
	Event(double epoch, long targetID, long projectileID, double relV, double mass);
	~Event();
	int GetEventType();
	string GetEventTypeString();
	double GetEventEpoch();
	long GetPrimary();
	long GetSecondary();
};

class DebrisPopulation
{
protected:
	long populationCount = 0;
	int scalingPower;
	double totalMass, currentEpoch, startEpoch, averageSemiMajorAxis, durationDays;
	map<long, DebrisObject> loadingPopulation;
	vector<pair<double, long>> initEpochs;
	int eventCount, explosionCount, collisionCount, collisionAvoidanceCount;
	//TODO - add count of object types

public:
	map<long, DebrisObject> population, removedPopulation;
	vector<Event> eventLog;

public:
	DebrisPopulation();
	~DebrisPopulation();
	void LoadPopulation();

	void Clear();
	double GetNextInitEpoch();
	double GetTimeToNextInitEpoch();
	double GetEpoch();
	int GetPopulationSize();
	int GetScalingPower();
	double GetAverageSMA();
	void SetDuration(double duration);
	double GetDuration();
	void SetAverageSMA(double averageSMA);
	void SetScalingPower(int power);
	void UpdateEpoch(double timeStep);
	void InitialiseEpoch(double epoch);
	void AddDebrisObject(DebrisObject debris);
	void AddDebrisEvent(Event debrisEvent);
	DebrisObject& GetObject(long ID);
	void DecayObject(long ID);
	void ExplodeObject(long ID);
	void CollideObject(long ID);
	int GetEventCount();
	int GetExplosionCount();
	int GetCollsionCount();
	int GetCAMCount();
	tuple<double, int, int, tuple<int, int, int>> GetPopulationState();
};

