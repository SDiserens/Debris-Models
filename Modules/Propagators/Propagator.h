#pragma once
class Propagator
{
public:
	Propagator();
	~Propagator();
	void PropagatePopulation(DebrisPopulation & population, double timestep); // timestep in seconds

protected:
	virtual OrbitalElements UpdateElements(OrbitalElements initialElements, double timeStep) = 0;
};

