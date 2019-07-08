#pragma once
#include "stdafx.h"
#include "..\Propagator.h"
#include"Implementation\sgp4io.h"


class SGP4 :
	public Propagator
{
public:
	SGP4();
	~SGP4();
};

