#pragma once
#include "stdafx.h"
#include "../Propagator.h"


class Analytic_J2 : public Propagator
{
protected:
	bool useJ2;
	const double J2term = 1.08262668e-3;  // J2 term for earths graviational field

public:
	Analytic_J2(DebrisPopulation & initPopulation, bool useJ2);
	~Analytic_J2();

protected:

	void UpdateElements(DebrisObject &object, double timeStep);
	
};

