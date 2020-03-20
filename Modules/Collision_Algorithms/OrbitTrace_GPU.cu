// OrbitTrace.cpp : contains the implementation of the GPU elements of the Orbit Trace collision algorithm.
//

#include "stdafx.h"
#include "OrbitTrace.h"

// CUDA standard includes
#include <windows.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "math.h"
#include "device_launch_parameters.h"
#include <thrust\for_each.h>
#include <thrust\execution_policy.h>
#include <thrust\device_vector.h>

#define CUDASTRIDE 256

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

__device__ double CollisionRate(CollisionPair &objectPair, double pAThreshold)
{
	double collisionRate, boundingRadii, minSeperation, relativeVelocity;
	vector3D velocityI, velocityJ;

	minSeperation = objectPair.CalculateMinimumSeparation();

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
__device__ bool HeadOnFilter(CollisionPair& objectPair)
{
	bool headOn = false;
	double deltaW;
	double eLimitP = objectPair.GetBoundingRadii() / objectPair.primary.GetElements().semiMajorAxis;
	double eLimitS = objectPair.GetBoundingRadii() / objectPair.secondary.GetElements().semiMajorAxis;
	// OT Head on filter
	if ((objectPair.primary.GetElements().eccentricity <= eLimitP) && (objectPair.secondary.GetElements().eccentricity <= eLimitS))
		headOn = true;
	else
	{
		deltaW = fabs((double)(Pi - objectPair.primary.GetElements().argPerigee - objectPair.secondary.GetElements().argPerigee));
		if (deltaW <= 1)
			headOn = true;
		else if (Tau - deltaW <= 1)
			headOn = true;
	}

	return headOn;
}

__device__ bool SynchronizedFilter(CollisionPair& objectPair, double timeStep)
{
	double meanMotionP, meanMotionS, driftAngle;
	// OT synch filter
	meanMotionP = Tau / objectPair.primary.GetPeriod();
	meanMotionS = Tau / objectPair.secondary.GetPeriod();

	driftAngle = fabs(meanMotionP - meanMotionS) * timeStep;
	return (driftAngle >= Tau);
}

__device__ bool ProximityFilter(CollisionPair& objectPair)
{
	//  OT  proximity filter
	double deltaMP, deltaMS, deltaMAngle, deltaMLinear, combinedSemiMajorAxis;
	OrbitalAnomalies anomaliesP, anomaliesS;

	anomaliesP.SetTrueAnomaly(objectPair.approachAnomalyP);
	anomaliesS.SetTrueAnomaly(objectPair.approachAnomalyS);

	deltaMP = fabs(anomaliesP.GetMeanAnomaly(objectPair.primary.GetElements().eccentricity) - objectPair.primary.GetElements().GetMeanAnomaly());
	deltaMS = fabs(anomaliesS.GetMeanAnomaly(objectPair.secondary.GetElements().eccentricity) - objectPair.secondary.GetElements().GetMeanAnomaly());

	combinedSemiMajorAxis = (objectPair.primary.GetElements().semiMajorAxis + objectPair.secondary.GetElements().semiMajorAxis) / 2;
	deltaMAngle = fabs(deltaMP - deltaMS);
	deltaMLinear = deltaMAngle * combinedSemiMajorAxis;

	return (deltaMLinear <= objectPair.GetBoundingRadii());
}


struct CollisionFilterKernel {
	double timeStep;
	CollisionFilterKernel(double timestep) {
		timeStep = timestep;
	}
	__device__ void operator()(CollisionPair& objectPair) {
		objectPair.collision = false;

		objectPair.CalculateRelativeInclination();
		double combinedSemiMajorAxis = objectPair.primary.GetElements().semiMajorAxis + objectPair.secondary.GetElements().semiMajorAxis;
		bool coplanar = objectPair.GetRelativeInclination() <= (2 * asin(objectPair.GetBoundingRadii() / combinedSemiMajorAxis));
		objectPair.coplanar = coplanar;

		if (coplanar)
		{
			// Calculate orbit intersections for coplanar
			objectPair.CalculateArgumenstOfIntersectionCoplanar();
			if (HeadOnFilter(objectPair) || !SynchronizedFilter(objectPair, timeStep) || ProximityFilter(objectPair))
				objectPair.collision = true;
		}
		else
		{
			// Calculate intersections for non coplanar
			objectPair.CalculateArgumenstOfIntersection();
			if (!SynchronizedFilter(objectPair, timeStep) || ProximityFilter(objectPair))
				objectPair.collision = true;
		}
	}
};

struct CollisionRateKernel {
		double timeStep, pAThreshold;
		CollisionRateKernel(double timestep, double threshold) {
			timeStep = timestep;
			pAThreshold = threshold;
		}
	__device__ void operator()(CollisionPair& objectPair) {
		if (objectPair.collision)
		{
			double  altitude, mass;
			thrust::pair<long, long> pairID;
			objectPair.probability = timeStep * CollisionRate(objectPair, pAThreshold);
			pairID = thrust::make_pair(objectPair.primaryID, objectPair.secondaryID);

			altitude = objectPair.primary.GetElements().GetRadialPosition();
			mass = objectPair.primary.GetMass() + objectPair.secondary.GetMass();
			objectPair.tempEvent = Event(0, pairID.first, pairID.second, objectPair.GetRelativeVelocity(), mass, altitude);
		}
		else
			objectPair.probability = 0;
	}
};

__host__ void OrbitTrace::MainCollision_GPU(DebrisPopulation & population, double timestep)
{
	double tempProbability, epoch = population.GetEpoch();
	thrust::device_vector<CollisionPair> pairList;
	int i, j, n;

	// Filter Cube List
	pairList = CreatePairList_GPU(population);
	timeStep = timestep;
	//unsigned int numThreads, numBlocks;
	//computeGridSize(pairList.size(), 256, numBlocks, numThreads);
	//TODO - Add code for GPU use
	n = pairList.size();
	for (i = 0; i < n; i += CUDASTRIDE) {
		j = min(i + CUDASTRIDE, n);
		thrust::for_each(thrust::device, pairList.begin()+i, pairList.begin() + j, CollisionFilterKernel(timestep));
	}
	for (i = 0; i < n; i += CUDASTRIDE) {
		j = min(i + CUDASTRIDE, n);
		thrust::for_each(thrust::device, pairList.begin() + i, pairList.begin() + j, CollisionRateKernel(timestep, pAThreshold));
	}
	//thrust::host_vector<CollisionPair> pairList_local = pairList;

	for (int i = 0; i < pairList.size(); i++) {
		CollisionPair objectPair = pairList[i];
		tempProbability = objectPair.probability;
		//	-- Determine if collision occurs through MC (random number generation)
		if (outputProbabilities && tempProbability > 0)
		{
			//	-- Store collision probability
			objectPair.tempEvent.SetEpoch(epoch);
			newCollisionProbabilities.push_back(tempProbability);
			newCollisionList.push_back(objectPair.tempEvent);

		}
		else
		{
			if (DetermineCollision(tempProbability))
			{
				// Store Collisions 
				objectPair.tempEvent.SetEpoch(epoch);
				newCollisionList.push_back(objectPair.tempEvent);
			}
		}
	}

	elapsedTime += timeStep;

}

