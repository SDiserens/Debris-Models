#pragma once
#include "stdafx.h"
#include "../Propagator.h"

const double J2term = 1.08262668e-3;  // J2 term for earths graviational field

class Analytic_J2 : public Propagator
{
public:
	Analytic_J2(DebrisPopulation & initPopulation, bool useJ2);
	~Analytic_J2();

protected:
	bool useJ2;

	void UpdateElements(DebrisObject &object, double timeStep);
};

