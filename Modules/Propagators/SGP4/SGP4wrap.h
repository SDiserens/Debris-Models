#pragma once
#include "stdafx.h"
#include "..\Propagator.h"
#include "SGP4Code\SGP4.h"

const double rhoZero = 0.1570;

const char opsMode = 'i'; // operation mode of the SGP4, either i (improved) or a (afspc).
const gravconsttype gravModel = wgs72; // which set of constants to use  wgs72, wgs84

class SGP4 :
	public Propagator
{
public:
	SGP4(DebrisPopulation & initPopulation);
	~SGP4();

private:
	void UpdateElements(DebrisObject &object, double timeStep);
	double CalculateBStar(DebrisObject &object);
	void HandleSPG4Error(DebrisObject &object);
};

