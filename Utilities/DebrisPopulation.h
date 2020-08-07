#pragma once

class DebrisPopulation
{
protected:
	long populationCount = 0;
	int scalingPower;
	double totalMass, currentEpoch, startEpoch, averageSemiMajorAxis, durationDays, launchCycle;
	unordered_map<long, DebrisObject> loadingPopulation;
	vector<pair<double, long>> initEpochs;
	int eventCount, explosionCount, collisionCount, collisionAvoidanceCount;
	int upperStageCount, spacecraftCount, debrisCount;
	vector<Event> eventLog;
	vector<Event> definedEvents;
	vector<DebrisObject> launchTraffic;

public:
	unordered_map<long, DebrisObject> population, removedPopulation;

public:
	DebrisPopulation();
	~DebrisPopulation();
	void LoadPopulation();
	void UpdateEpoch(double timeStep);
	void InitialiseEpoch(double epoch);
	void LoadLaunchTraffic();
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
	int GetCAMCount();
	double GetDuration();
	DebrisObject& GetObject(long ID);

	void AddDebrisObject(DebrisObject debris);
	void AddDefinedEvent(Event breakup);
	void AddLaunchTraffic(vector<DebrisObject> launchTraffic, double launchCycle);
	void AddDebrisEvent(Event debrisEvent);
	void SetDuration(double duration);
	void SetAverageSMA(double averageSMA);
	void SetScalingPower(int power);

	vector<Event> GenerateExplosionList();
	vector<Event> GenerateDefinedEventList();
	void DecayObject(long ID);
	void ExplodeObject(long ID);
	void CollideObject(long ID);
	void RemoveObject(long ID, int type); // type: (0 = explosion; 1 = collision; 2 = decay)

	tuple<double, int, tuple<int, int, int>, int, tuple<int, int, int>> GetPopulationState();
	vector<Event> GetEventLog();
};

