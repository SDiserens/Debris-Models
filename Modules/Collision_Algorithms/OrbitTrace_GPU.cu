// OrbitTrace.cpp : contains the implementation of the GPU elements of the Orbit Trace collision algorithm.
//

#include "stdafx.h"
#include "OrbitTrace.h"

// CUDA standard includes
#include <windows.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "device_launch_parameters.h"
#include <thrust\for_each.h>
#include <thrust\execution_policy.h>
#include <thrust\device_vector.h>

//Round a / b to nearest higher integer value
unsigned int  iDivUp(unsigned int  a, unsigned int  b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

// compute grid and thread block size for a given number of elements
void computeGridSize(unsigned int n, unsigned int blockSize, unsigned int  &numBlocks, unsigned int  &numThreads)
{
	numThreads = min(blockSize, n);
	numBlocks = iDivUp(n, numThreads);
}

__device__ double OrbitTrace::CollisionRate(CollisionPair &objectPair)
{
	double collisionRate, boundingRadii, minSeperation, relativeVelocity;
	vector3D velocityI, velocityJ;

	switch (MOIDtype) {
	case 0: minSeperation = objectPair.CalculateMinimumSeparation();
	case 1: minSeperation = objectPair.CalculateMinimumSeparation_DL();
	case 2: minSeperation = objectPair.CalculateMinimumSeparation_MOID();
	}


	velocityI = objectPair.primary.GetVelocity();
	velocityJ = objectPair.secondary.GetVelocity();

	relativeVelocity = velocityI.CalculateRelativeVector(velocityJ).vectorNorm();
	boundingRadii = max(pAThreshold, objectPair.GetBoundingRadii());
	objectPair.SetRelativeVelocity(relativeVelocity);
	//sinAngle = velocityI.VectorCrossProduct(velocityJ).vectorNorm() / (velocityI.vectorNorm() * velocityJ.vectorNorm());

	// OT collision rate
	if (boundingRadii > minSeperation)
		collisionRate = Pi * boundingRadii * relativeVelocity /
		(2 * velocityI.VectorCrossProduct(velocityJ).vectorNorm()  * objectPair.primary.GetPeriod() * objectPair.secondary.GetPeriod());
	else
		collisionRate = 0;

	return collisionRate;
}
__host__ thrust::device_vector<CollisionPair> OrbitTrace::CreatePairList_GPU(DebrisPopulation & population)
{
	thrust::device_vector<CollisionPair> pairList;
	// TODO - GPU code for creating pairList
	//Talk to Pete about i, j where i < j < N
	for (auto it = population.population.begin(); it != population.population.end(); it++)
	{
		// For each subsequent object
		auto jt = it;
		for (++jt; jt != population.population.end(); ++jt)
		{
			/// Add pair to list
			//DebrisObject& primaryObject(population.Ge), secondaryObject;
			CollisionPair pair(it->second, jt->second);
			if (PerigeeApogeeTest(pair))
				pairList.push_back(pair);
			else
				pair.~CollisionPair();
		}
	}

	return pairList;
}

/*
//TODO - add device function  to operate on each collision pair
__device__ void OrbitTraceAlgorithm(list<CollisionPair>& pairList) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
}

__host__ void OrbitTrace::MainCollision_GPU_Cuda(DebrisPopulation & population, double timestep)
{
	double tempProbability, collisionRate, altitude, mass;
	list<CollisionPair> pairList;
	pair<long, long> pairID;
	bool collision;

	// Filter Cube List
	pairList = CreatePairList_GPU(population);
	timeStep = timestep;
	unsigned int numThreads, numBlocks;
	computeGridSize(pairList.size(), 256, numBlocks, numThreads);
	//TODO - Add code for GPU use
		// 1D iteration over pairList
		//int index = blockIdx.x * blockDim.x + threadIdx.x;

}
*/
__host__ __device__ void OrbitTrace::MainCollision_GPU(DebrisPopulation & population, double timestep)
{
	double tempProbability, collisionRate, altitude, mass;
	thrust::device_vector<CollisionPair> pairList;
	pair<long, long> pairID;
	bool collision;

	// Filter Cube List
	pairList = CreatePairList_GPU(population);
	timeStep = timestep;
	unsigned int numThreads, numBlocks;
	computeGridSize(pairList.size(), 256, numBlocks, numThreads);
	//TODO - Add code for GPU use
	thrust::for_each(thrust::device, pairList.begin(), pairList.end(), [&](CollisionPair& objectPair)
	{
		collision = false;

		objectPair.CalculateRelativeInclination();

		if (CoplanarFilter(objectPair))
		{
			// Calculate orbit intersections for coplanar
			objectPair.CalculateArgumenstOfIntersectionCoplanar();
			if (HeadOnFilter(objectPair) || !SynchronizedFilter(objectPair) || ProximityFilter(objectPair))
				collision = true;
		}
		else
		{
			// Calculate intersections for non coplanar
			objectPair.CalculateArgumenstOfIntersection();
			if (!SynchronizedFilter(objectPair) || ProximityFilter(objectPair))
				collision = true;
		}

		if (collision)
		{
			collisionRate = CollisionRate(objectPair);
			objectPair.probability = timeStep * collisionRate;
			pairID = make_pair(objectPair.primaryID, objectPair.secondaryID);

			altitude = objectPair.primary.GetElements().GetRadialPosition();
			mass = objectPair.primary.GetMass() + objectPair.secondary.GetMass();
			objectPair.tempEvent = Event(population.GetEpoch(), pairID.first, pairID.second, objectPair.GetRelativeVelocity(), mass, altitude);
		}
	});

	for (int i = 0; i < pairList.size(); i++) {
		CollisionPair objectPair = pairList[i];
		tempProbability = objectPair.probability;
		//	-- Determine if collision occurs through MC (random number generation)
		if (outputProbabilities && tempProbability > 0)
		{
			//	-- Store collision probability
			newCollisionProbabilities.push_back(tempProbability);
			newCollisionList.push_back(objectPair.tempEvent);

		}
		else
		{
			if (DetermineCollision(tempProbability))
			{
				// Store Collisions 
				newCollisionList.push_back(objectPair.tempEvent);
			}
		}
	}

	elapsedTime += timeStep;

}

