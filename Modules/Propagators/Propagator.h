#pragma once
class Propagator
{
public:
	Propagator(DebrisPopulation & initPopulation);
	~Propagator();
	void PropagatePopulation(double timestep); // timestep in seconds
	void SyncPopulation(); // timestep in seconds

protected:
	DebrisPopulation & population;

	vector<long> removeID;
	virtual void UpdateElements(DebrisObject &object, double timeStep) = 0;
	//virtual void HandleError(DebrisObject &object) = 0;

	void RemovePopulation();
	double CalculateBStar(DebrisObject &object);
};

