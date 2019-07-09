#pragma once
#include "stdafx.h"
#include "..\Propagator.h"
#include "SGP4Code\SGP4.h"

const double rhoZero = 0.1570;


class SGP4 :
	public Propagator
{
public:
	SGP4();
	~SGP4();

private:
	void UpdateElements(DebrisObject &object, double timeStep);
};

