// CUBE.cpp : contains the implementation of the CUBE algorithm.
//

#include "stdafx.h"
#include "CUBE.h"

CUBEApproach::CUBEApproach(double dimension, bool probabilities)
{
	cubeDimension = dimension;
	cubeVolume = dimension * dimension * dimension;
	outputProbabilities = probabilities;
}

void CUBEApproach::mainCollision(DebrisPopulation& population, double timeStep)
{
	

	// For each object in population -
	//	-- Generate Mean anomaly (randomTau)
	//	-- Calculate position
	//	-- Identify CUBE ID

	// Filter CUBEIDs
	//	-- Hash ID
	//	-- Sort Hash list
	//	-- Identify duplicate hash values
	//	-- (Sanitize results to remove hash clashes)

	// For each conjunction (cohabiting pair)
	//	-- Calculate collision rate in cube
	//	-- Either
	//		-- Store collision probability
	//	-- OR
	//		-- Determine if collision occurs through MC (random number generation)

	// Store Collisions 

}