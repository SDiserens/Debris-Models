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
#include <thrust\remove.h>

#define CUDASTRIDE 256
typedef thrust::device_vector<CollisionPair>::iterator dvit;

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

__host__ thrust::host_vector<CollisionPair> OrbitTrace::CreatePairList_GPU(DebrisPopulation & population)
{
	thrust::host_vector<CollisionPair> pairList;
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
__device__ bool HeadOnFilter(CollisionPair objectPair)
{
	bool headOn = false;
	double deltaW;
	double eLimitP = objectPair.GetBoundingRadii() / objectPair.primaryElements.semiMajorAxis;
	double eLimitS = objectPair.GetBoundingRadii() / objectPair.secondaryElements.semiMajorAxis;
	// OT Head on filter
	if ((objectPair.primaryElements.eccentricity <= eLimitP) && (objectPair.secondaryElements.eccentricity <= eLimitS))
		headOn = true;
	else
	{
		deltaW = fabs((double)(Pi - objectPair.primaryElements.argPerigee - objectPair.secondaryElements.argPerigee));
		if (deltaW <= 1)
			headOn = true;
		else if (Tau - deltaW <= 1)
			headOn = true;
	}

	return headOn;
}

__device__ bool SynchronizedFilter(CollisionPair objectPair, double timeStep)
{
	double meanMotionP, meanMotionS, driftAngle;
	// OT synch filter
	meanMotionP = Tau / objectPair.primaryElements.CalculatePeriod();
	meanMotionS = Tau / objectPair.secondaryElements.CalculatePeriod();

	driftAngle = fabs(meanMotionP - meanMotionS) * timeStep;
	return (driftAngle >= Tau);
}

__device__ bool ProximityFilter(CollisionPair objectPair)
{
	//  OT  proximity filter
	double deltaMP, deltaMS, deltaMAngle, deltaMLinear, combinedSemiMajorAxis;
	OrbitalAnomalies anomaliesP, anomaliesS;

	anomaliesP.SetTrueAnomaly(objectPair.approachAnomalyP);
	anomaliesS.SetTrueAnomaly(objectPair.approachAnomalyS);

	deltaMP = fabs(anomaliesP.GetMeanAnomaly(objectPair.primaryElements.eccentricity) - objectPair.primaryElements.GetMeanAnomaly());
	deltaMS = fabs(anomaliesS.GetMeanAnomaly(objectPair.secondaryElements.eccentricity) - objectPair.secondaryElements.GetMeanAnomaly());

	combinedSemiMajorAxis = (objectPair.primaryElements.semiMajorAxis + objectPair.secondaryElements.semiMajorAxis) / 2;
	deltaMAngle = fabs(deltaMP - deltaMS);
	deltaMLinear = deltaMAngle * combinedSemiMajorAxis;

	return (deltaMLinear <= objectPair.GetBoundingRadii());
}

struct NotCollision
{
	__host__ __device__
		bool operator()(CollisionPair objectPair)
	{
		return (!objectPair.collision);
	}
};

struct CollisionFilterKernel {
	double timeStep;
	CollisionFilterKernel(double timestep) {
		timeStep = timestep;
	}
	__device__ void operator()(CollisionPair &objectPair) const{
		objectPair.collision = false;

		double combinedSemiMajorAxis = objectPair.primaryElements.semiMajorAxis + objectPair.secondaryElements.semiMajorAxis;
		bool coplanar = objectPair.relativeInclination <= (2 * asin(objectPair.boundingRadii / combinedSemiMajorAxis));
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

__device__ double CollisionRate(CollisionPair &objectPair, double pAThreshold)
{
	double collisionRate, boundingRadii, relativeVelocity;


	boundingRadii = max(pAThreshold, objectPair.GetBoundingRadii());
	vector3D velocityI, velocityJ;
	velocityI = objectPair.primaryElements.GetVelocity();
	velocityJ = objectPair.secondaryElements.GetVelocity();

	relativeVelocity = velocityI.CalculateRelativeVector(velocityJ).vectorNorm();
	objectPair.SetRelativeVelocity(relativeVelocity);
	//sinAngle = velocityI.VectorCrossProduct(velocityJ).vectorNorm() / (velocityI.vectorNorm() * velocityJ.vectorNorm());

	// OT collision rate
	if (boundingRadii > objectPair.minSeperation)
		collisionRate = Pi * boundingRadii * relativeVelocity /
		(2 * velocityI.VectorCrossProduct(velocityJ).vectorNorm()  * objectPair.primaryElements.CalculatePeriod() * objectPair.secondaryElements.CalculatePeriod());
	else
		collisionRate = 0;
		objectPair.collision = false;

	return collisionRate;
}
struct MinSeperation {
	MinSeperation() {};
	__device__ void operator()(CollisionPair &objectPair) {
		objectPair.minSeperation = objectPair.CalculateMinimumSeparation();

		objectPair.altitude = objectPair.primaryElements.GetRadialPosition();

	}
};

struct CollisionRateKernel {
		double timeStep, pAThreshold;
		CollisionRateKernel(double timestep, double threshold) {
			timeStep = timestep;
			pAThreshold = threshold;
		}
		__device__ void operator()(CollisionPair &objectPair) {
			objectPair.probability = timeStep * CollisionRate(objectPair, pAThreshold);
		}
};

__host__ void OrbitTrace::MainCollision_GPU(DebrisPopulation & population, double timestep)
{
	double mass, tempProbability, epoch = population.GetEpoch();
	Event tempEvent;
	thrust::host_vector<CollisionPair> hostList(0);

	// Filter Cube List
	hostList = CreatePairList_GPU(population);
	timeStep = timestep;
	//unsigned int numThreads, numBlocks;
	//computeGridSize(pairList.size(), 256, numBlocks, numThreads);
	//TODO - Add code for GPU use
	thrust::device_vector<CollisionPair> pairList = hostList;
	thrust::for_each(thrust::device, pairList.begin(), pairList.begin(), CollisionFilterKernel(timestep));
	
	dvit collisionEnd = thrust::remove_if(thrust::device, pairList.begin(), pairList.end(), NotCollision());
	//pairList.erase(collisionEnd, pairList.end());

	thrust::for_each(thrust::device, pairList.begin(), collisionEnd, MinSeperation());
	thrust::for_each(thrust::device, pairList.begin(), collisionEnd, CollisionRateKernel(timestep, pAThreshold));

	collisionEnd = thrust::remove_if(thrust::device, pairList.begin(), pairList.end(), NotCollision());

	thrust::host_vector<CollisionPair> outList(pairList.begin(), collisionEnd);

	for (int i = 0; i < outList.size(); i++) {
		CollisionPair objectPair = outList[i];
		tempProbability = objectPair.probability;

		mass = objectPair.primaryMass + objectPair.secondaryMass;
		tempEvent = Event(epoch, objectPair.primaryID, objectPair.secondaryID, objectPair.GetRelativeVelocity(), mass, objectPair.altitude);
		//	-- Determine if collision occurs through MC (random number generation)
		if (outputProbabilities && tempProbability > 0)
		{
			//	-- Store collision probability
			newCollisionProbabilities.push_back(tempProbability);
			newCollisionList.push_back(tempEvent);

		}
		else
		{
			if (DetermineCollision(tempProbability))
			{
				// Store Collisions 
				newCollisionList.push_back(tempEvent);
			}
		}
	}

	elapsedTime += timeStep;

}

