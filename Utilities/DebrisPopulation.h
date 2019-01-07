#pragma once
class DebrisPopulation
{
	long populationCount;
	float totalMass;
	std::vector<DebrisObject> population;

public:
	DebrisPopulation();
	~DebrisPopulation();
	void AddDebrisObject(DebrisObject debris);
};

