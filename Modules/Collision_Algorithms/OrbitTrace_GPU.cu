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


struct PairKernel {
	int n;
	DebrisObject * population;
	PairKernel(int numObjects, DebrisObject *_data) {
		n = numObjects;
		population = _data;
	}
	template <typename Tuple>
	__device__ void operator()(Tuple t) {
		int x, y, z, i;
		i = thrust::get<0>(t);
		z = n - 1;
		x = 1;
		while (i > z) {
			i -= z;
			--z;
			++x;
		}
		y = x + i;

		thrust::get<1>(t).SetCollisionPair(population[--x], population[--y]);
	}
};

struct PAKernel {
	double pAThreshold;
	PAKernel(double threshold) {
		pAThreshold = threshold;
	}
	__device__ bool operator()(CollisionPair objectPair) {
		double maxPerigee, minApogee;
		// Perigee Apogee Test
		maxPerigee = max(objectPair.primaryElements.GetPerigee(), objectPair.secondaryElements.GetPerigee());
		minApogee = min(objectPair.primaryElements.GetApogee(), objectPair.secondaryElements.GetApogee());

		return (maxPerigee - minApogee) <= max(pAThreshold, objectPair.GetBoundingRadii());
	}
};

__host__ thrust::device_vector<CollisionPair> OrbitTrace::CreatePairList_GPU(DebrisPopulation & population)
{
	int n = population.GetPopulationSize();
	int N = n*(n - 1) / 2;
	thrust::device_vector<CollisionPair> pairList(N);
	thrust::device_vector<DebrisObject> populationList;
	
	for_each(population.population.begin(), population.population.end(), [&](pair<long, DebrisObject> object) {
		populationList.push_back(object.second);
	});

	thrust::counting_iterator<int> first(1);
	thrust::counting_iterator<int> last = first + N;

	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(first, pairList.begin())), thrust::make_zip_iterator(thrust::make_tuple(last, pairList.end())), PairKernel(n, thrust::raw_pointer_cast(populationList.data())));

	int p = thrust::copy_if(thrust::device, pairList.begin(), pairList.end(), pairList.begin(), PAKernel(pAThreshold)) - pairList.begin();

	populationList.clear();
	populationList.shrink_to_fit();

	pairList.resize(p);
	pairList.shrink_to_fit();

	return pairList;
}

__host__ thrust::device_vector<CollisionPair> OrbitTrace::CreatePairList_CPU(DebrisPopulation & population)
{

	thrust::device_vector<CollisionPair> pairList;
	mutex mtx;
	concurrency::parallel_for_each(population.population.begin(), population.population.end(), [&](auto& it) {
		auto jt = population.population.find(it.first);
		for (++jt; jt != population.population.end(); ++jt)
		{
			/// Add pair to list
			//DebrisObject& primaryObject(population.Ge), secondaryObject;
			CollisionPair pair(it.second, jt->second);
			if (PerigeeApogeeTest(pair)) {
				mtx.lock();
				pairList.push_back(pair);
				mtx.unlock();
			}
			else
				pair.~CollisionPair();
		}
	});
	return pairList;
}

/*
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
	//- Add code for GPU use
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

struct Collision
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
	__device__ void operator()(CollisionPair &objectPair) {
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
			if (objectPair.collision) {
			}
		}
	}
};
struct LowerBoundFilter {
	double pAThreshold;
	LowerBoundFilter(double threshold) {
		pAThreshold = threshold;
	}
	__device__ void operator()(CollisionPair &objectPair) {
		if (objectPair.CalculateLowerBoundSeparation() > max(pAThreshold, objectPair.boundingRadii))
		{
			objectPair.collision = false;
		}
	}
};
struct MinSeperation {
	MinSeperation() {};
	__device__ void operator()(CollisionPair &objectPair) {
		double sep = objectPair.CalculateMinimumSeparation();

		//return objectPair.minSeperation;
	}
};

struct CollisionRateKernel {
		double timeStep, pAThreshold;
		bool relativeGravity;
		CollisionRateKernel(double timestep, double threshold, bool relG) {
			timeStep = timestep;
			pAThreshold = threshold;
			relativeGravity = relG;
		}
		__device__ double operator()(CollisionPair &objectPair) {

			double collisionRate, threshold, boundingRadii, relativeVelocity, scaling, escapeVelocity2, gravitationalPerturbation, collisionRate2;
			
			boundingRadii = objectPair.GetBoundingRadii();
			vector3D velocityI, velocityJ;
			boundingRadii = objectPair.GetBoundingRadii();
			threshold = max(pAThreshold, boundingRadii);
			//sinAngle = velocityI.VectorCrossProduct(velocityJ).vectorNorm() / (velocityI.vectorNorm() * velocityJ.vectorNorm());
			scaling = 1;
			if (boundingRadii < pAThreshold) {
				scaling = boundingRadii / pAThreshold;
				scaling = scaling * scaling;
			}
			if (relativeGravity)
			{
				escapeVelocity2 = 2 * (objectPair.primaryMass + objectPair.secondaryMass) * GravitationalConstant / boundingRadii;
			}

			// OT collision rate
			if (objectPair.minSeperation < threshold)
			{
				objectPair.primaryElements.SetTrueAnomaly(objectPair.approachAnomalyP);
				objectPair.secondaryElements.SetTrueAnomaly(objectPair.approachAnomalyS);
				velocityI = objectPair.primaryElements.GetVelocity();
				velocityJ = objectPair.secondaryElements.GetVelocity();
				relativeVelocity = objectPair.relativeVelocity = velocityI.CalculateRelativeVector(velocityJ).vectorNorm();
				if (relativeGravity)
				{
					gravitationalPerturbation = (1 + escapeVelocity2 / (relativeVelocity * relativeVelocity));
				}
				else
					gravitationalPerturbation = 1;

				collisionRate = gravitationalPerturbation * Pi * threshold * relativeVelocity /
					(2 * velocityI.VectorCrossProduct(velocityJ).vectorNorm()  * objectPair.primaryElements.CalculatePeriod() * objectPair.secondaryElements.CalculatePeriod());
			}
			else
				collisionRate = 0;

			if (objectPair.minSeperation2 < threshold)
			{

				objectPair.primaryElements.SetTrueAnomaly(objectPair.approachAnomalyP2);
				objectPair.secondaryElements.SetTrueAnomaly(objectPair.approachAnomalyS2);
				velocityI = objectPair.primaryElements.GetVelocity();
				velocityJ = objectPair.secondaryElements.GetVelocity();
				relativeVelocity = objectPair.relativeVelocity2 = velocityI.CalculateRelativeVector(velocityJ).vectorNorm();
				if (relativeGravity)
				{
					gravitationalPerturbation = (1 + escapeVelocity2 / (relativeVelocity * relativeVelocity));
				}
				else
					gravitationalPerturbation = 1;

				collisionRate2 = gravitationalPerturbation * Pi * threshold * relativeVelocity /
					(2 * velocityI.VectorCrossProduct(velocityJ).vectorNorm()  * objectPair.primaryElements.CalculatePeriod() * objectPair.secondaryElements.CalculatePeriod());
			}
			else
				collisionRate2 = 0;
 
			objectPair.probability = timeStep * scaling * (collisionRate + collisionRate2);
			return objectPair.probability;
		}
};

__host__ void OrbitTrace::MainCollision_GPU(DebrisPopulation & population, double timestep)
{
	double mass, tempProbability, epoch = population.GetEpoch(), sep;
	Event tempEvent;

	// Filter Cube List
	thrust::device_vector<CollisionPair> pairList = CreatePairList_GPU(population);
	timeStep = timestep;
	//unsigned int numThreads, numBlocks;
	//computeGridSize(pairList.size(), 256, numBlocks, numThreads);
	size_t n = pairList.size();
	thrust::for_each(thrust::device, pairList.begin(), pairList.end(), CollisionFilterKernel(timestep));
	
	n = thrust::remove_if(pairList.begin(), pairList.end(), Collision()) - pairList.begin();
	pairList.resize(n);

	thrust::for_each(thrust::device, pairList.begin(), pairList.end(), LowerBoundFilter(pAThreshold));
	n = thrust::remove_if(pairList.begin(), pairList.end(), Collision()) - pairList.begin();
	pairList.resize(n);

	thrust::host_vector<CollisionPair> outList(pairList.begin(), pairList.end());
	pairList.clear();
	pairList.shrink_to_fit();
	//n = std::remove_if(outList.begin(), outList.end(), Collision()) - outList.begin();
	//outList.resize(n);

	concurrency::parallel_for_each(outList.begin(), outList.end(), [&](CollisionPair& objectPair)
	{	
		switch (MOIDtype) {
		case 0:
			sep = objectPair.CalculateMinimumSeparation();
			break;
		case 1:
			sep = objectPair.CalculateMinimumSeparation_DL(max_root_error, min_root_error, max_anom_error);
			break;
		case 2:
			sep = objectPair.CalculateMinimumSeparation_MOID();
			break;
		}
	});

	pairList = thrust::device_vector<CollisionPair>(outList.begin(), outList.end());
	
	//thrust::for_each(thrust::device, pairList.begin(), pairList.end(), MinSeperation());

	thrust::device_vector<double> probabilityList(n);
	thrust::transform(thrust::device, pairList.begin(), pairList.end(), probabilityList.begin(), CollisionRateKernel(timestep, pAThreshold, relativeGravity));

	//dvit collisionEnd = thrust::remove_if(thrust::device, pairList.begin(), pairList.end(), NotCollision());

	outList = thrust::host_vector<CollisionPair>(pairList.begin(), pairList.end());
	//thrust::host_vector<CollisionPair> outList(pairList.begin(), pairList.end());
	thrust::host_vector<double> pList(probabilityList.begin(), probabilityList.end());
	pairList.clear();
	pairList.shrink_to_fit();
	probabilityList.clear();
	probabilityList.shrink_to_fit();
	//thrust::host_vector<double> probOut(probabilityList.begin(), probabilityList.end());

	for (int i = 0; i < outList.size(); i++) {
		CollisionPair objectPair = outList[i];
		tempProbability = pList[i];
		if (tempProbability > 0) {
			mass = objectPair.primaryMass + objectPair.secondaryMass;
			tempEvent = Event(epoch, objectPair.primaryID, objectPair.secondaryID, objectPair.GetRelativeVelocity(), mass, objectPair.GetCollisionAltitude(), objectPair.GetMinSeparation(), tempProbability);
			tempEvent.SetCollisionAnomalies(objectPair.approachAnomalyP, objectPair.approachAnomalyS);
			// Store Collisions 
			newCollisionList.push_back(tempEvent);
		}
	}
	outList.clear();
	outList.shrink_to_fit();
	pList.clear();
	pList.shrink_to_fit();

	elapsedTime += timeStep;

}

