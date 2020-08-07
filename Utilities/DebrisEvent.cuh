class Event
{

public:
	static int eventSEQ;
	long eventID, debrisGenerated, primaryID, secondaryID;
	int eventType; // ( 0 : Explosion, 1 : Collision,  2 : Collision Avoidance,)
	double eventEpoch, altitude, involvedMass, relativeVelocity, energyMassRatio, primaryAnomaly, secondaryAnomaly, minSeparation, collisionProbability;
	bool catastrophic, momentumConserved;

public:
	CUDA_CALLABLE_MEMBER Event();
	CUDA_CALLABLE_MEMBER Event(double epoch, long objectID, double mass);
	CUDA_CALLABLE_MEMBER Event(double epoch, long objectID, bool consMomentum, bool catastr, double mass, long debrisCount);
	CUDA_CALLABLE_MEMBER Event(double epoch, long targetID, long projectileID, double relV, double mass, double alt, double separation, double probability);
	CUDA_CALLABLE_MEMBER ~Event();
	void CollisionAvoidance();
	void SwapPrimarySecondary();
	void SetEpoch(double epoch);
	void SetEventID();
	void SetAltitude(double alt);
	void SetCollisionAnomalies(double pimaryM, double secondaryM);
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
