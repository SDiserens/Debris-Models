#pragma once

static int objectSEQ = 0;

class DebrisPopulation
{
protected:
	long populationCount = 0;
	int scalingPower;
	double totalMass, currentEpoch, startEpoch, averageSemiMajorAxis, durationDays;
	unordered_map<long, DebrisObject> loadingPopulation;
	vector<pair<double, long>> initEpochs;
	int eventCount, explosionCount, collisionCount, collisionAvoidanceCount, falseCollisionCount, catCollisionCount;
	int upperStageCount, spacecraftCount, debrisCount;
	bool launches;
	vector<Event> eventLog;
	vector<Event> definedEvents;
	vector<DebrisObject> launchTraffic;

public:
	unordered_map<long, DebrisObject> population;
	unordered_map<long, RemovedObject> removedPopulation;

public:
	DebrisPopulation();
	~DebrisPopulation();
	void LoadPopulation();
	void UpdateEpoch(double timeStep);
	void InitialiseEpoch(double epoch);
	void LoadLaunchTraffic(int n);
	void Clear();

	double GetNextInitEpoch();
	double GetTimeToNextInitEpoch();
	double GetEpoch();
	double GetStartEpoch();
	int GetPopulationSize();
	int GetScalingPower();
	double GetAverageSMA();
	int GetEventCount();
	int GetExplosionCount();
	int GetCollsionCount();
	int GetCatCollsionCount();
	int GetCAMCount();
	double GetDuration();
	DebrisObject& GetObject(long ID);
	bool CheckObject(long ID);
	bool CheckLaunches();

	long AddDebrisObject(DebrisObject debris);
	void AddDefinedEvent(Event breakup);
	void AddLaunchTraffic(vector<DebrisObject> launchTraffic);
	void AddDebrisEvent(Event debrisEvent);
	void SetDuration(double duration);
	void SetAverageSMA(double averageSMA);
	void SetScalingPower(int power);
	void SetLaunches(bool launch);

	vector<Event> GenerateExplosionList();
	vector<Event> GenerateDefinedEventList();
	void DecayObject(long ID);
	void ExplodeObject(long ID);
	void CollideObject(long ID);
	void RemoveObject(long ID, int type); // type: (0 = explosion; 1 = collision; 2 = decay)

	tuple<double, int, tuple<int, int, int>, int, tuple<int, int, int, int>> GetPopulationState();
	vector<Event> GetEventLog();
};


DebrisObject CopyDebrisObject(DebrisObject & object);