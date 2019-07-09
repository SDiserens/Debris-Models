#pragma once
class Propagator
{
public:
	Propagator();
	~Propagator();
	void PropagatePopulation(DebrisPopulation & population, double timestep); // timestep in seconds

protected:
	virtual void UpdateElements(DebrisObject &object, double timeStep) = 0;
};

