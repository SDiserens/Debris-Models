#pragma once
#include "stdafx.h"
#include "..\Propagator.h"
#include "SGP4Code\SGP4.h"

class SGP4 :
	public Propagator
{
private:
	char opsMode = 'a'; // operation mode of the SGP4, either i (improved) or a (afspc).
	gravconsttype gravModel = wgs72old; // which set of constants to use  wgs72, wgs84

public:
	SGP4(DebrisPopulation & initPopulation);
	SGP4(DebrisPopulation & initPopulation, char opMode, gravconsttype gravMode);
	~SGP4();

private:
	void UpdateElements(DebrisObject &object, double timeStep);
	void HandleError(DebrisObject &object);
};

