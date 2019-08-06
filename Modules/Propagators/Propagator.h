#pragma once
class Propagator
{
public:
	Propagator(DebrisPopulation & initPopulation);
	~Propagator();
	void PropagatePopulation(double timestep); // timestep in seconds

protected:
	DebrisPopulation & population;
	virtual void UpdateElements(DebrisObject &object, double timeStep) = 0;
	//virtual void HandleError(DebrisObject &object) = 0;

	double CalculateBStar(DebrisObject &object);
};

