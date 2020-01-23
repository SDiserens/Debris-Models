#pragma once
class Event
{

public:
	static int eventSEQ;
	long eventID, debrisGenerated, primaryID, secondaryID;
	int eventType; // ( 0 : Explosion, 1 : Collision,  2 : Collision Avoidance,)
	double eventEpoch, altitude, involvedMass, relativeVelocity, energyMassRatio;
	bool catastrophic, momentumConserved;

public:
	Event();
	Event(double epoch, long objectID, double mass);
	Event(double epoch, long objectID, bool consMomentum, bool catastr, double mass, long debrisCount);
	Event(double epoch, long targetID, long projectileID, double relV, double mass, double alt);
	~Event();
	void CollisionAvoidance();
	void SwapPrimarySecondary();
	void SetConservationMomentum(bool conservedFlag);
	void SetCatastrophic(bool catastrophicFlag);
	void SetEMR(double eMRatio);
	void SetDebrisCount(long count);
	int GetEventType();
	string GetEventTypeString();
	double GetEventEpoch();
	long GetPrimary();
	long GetSecondary();
	pair<long, long> GetCollisionPair();
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
	int upperStageCount, spacecraftCount, debrisCount;
	vector<Event> eventLog;

public:
	map<long, DebrisObject> population, removedPopulation;

public:
	DebrisPopulation();
	~DebrisPopulation();
	void LoadPopulation();
	void UpdateEpoch(double timeStep);
	void InitialiseEpoch(double epoch);
	void Clear();

	double GetNextInitEpoch();
	double GetTimeToNextInitEpoch();
	double GetEpoch();
	int GetPopulationSize();
	int GetScalingPower();
	double GetAverageSMA();
	int GetEventCount();
	int GetExplosionCount();
	int GetCollsionCount();
	int GetCAMCount();
	double GetDuration();
	DebrisObject& GetObject(long ID);

	void AddDebrisObject(DebrisObject debris);
	void AddDebrisEvent(Event debrisEvent);
	void SetDuration(double duration);
	void SetAverageSMA(double averageSMA);
	void SetScalingPower(int power);

	vector<Event> GenerateExplosionList();
	void DecayObject(long ID);
	void ExplodeObject(long ID);
	void CollideObject(long ID);
	void RemoveObject(long ID, int type); // type: (0 = explosion; 1 = collision; 2 = decay)

	tuple<double, int, tuple<int, int, int>, int, tuple<int, int, int>> GetPopulationState();
	vector<Event> GetEventLog();
};

