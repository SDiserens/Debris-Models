// HootsFilter_GPU.cu : contains the implementation of the GPU elements of the Hoots collision algorithm.
//

#include "stdafx.h"
#include "HootsFilter.h"

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

__host__ thrust::device_vector<CollisionPair> HootsFilter::CreatePairList_GPU(DebrisPopulation & population)
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
	pairList.resize(p);

	return pairList;
}


struct Collision
{
	__host__ __device__
		bool operator()(CollisionPair objectPair)
	{
		return (objectPair.collision);
	}
};

struct GeometricFilterKernel {
	double threshold;
	GeometricFilterKernel(double threshold) {
		threshold = threshold;
	}
	__device__ bool operator()(CollisionPair &objectPair) {
		objectPair.collision = (objectPair.minSeperation < threshold);

		return objectPair.collision;
	}
};

struct TimeFilterKernel {
	double timeStep;
	TimeFilterKernel(double timestep) {
		timeStep = timestep;
	}
	__device__ bool operator()(CollisionPair &objectPair) {
		objectPair.collision = false;


		return objectPair.collision;
	}
};

void HootsFilter::MainCollision_GPU(DebrisPopulation & population, double timestep)
{
	double mass, tempProbability, epoch = population.GetEpoch();
	Event tempEvent;
	pair<long, long> pairID;
	vector<double> candidateTimeList, collisionTimes;
	double closeTime, closeApproach;
	bool collide;

	// Filter Cube List
	thrust::device_vector<CollisionPair> pairListIn = CreatePairList_GPU(population);
	timeStep = timestep;
	//unsigned int numThreads, numBlocks;
	//computeGridSize(pairList.size(), 256, numBlocks, numThreads);
	size_t n = pairListIn.size();
	thrust::device_vector<CollisionPair> pairList(n);

	thrust::host_vector<CollisionPair> outList(pairList.begin(), pairList.end());
	concurrency::parallel_for_each(outList.begin(), outList.end(), [&](CollisionPair& objectPair)
	{	switch (MOIDtype) {
	case 0:
		objectPair.minSeperation = objectPair.CalculateMinimumSeparation();
		break;
	case 1:
		objectPair.minSeperation = objectPair.CalculateMinimumSeparation_DL();
		break;
	case 2:
		objectPair.minSeperation = objectPair.CalculateMinimumSeparation_MOID();
		break;
	}
	});

	pairList = thrust::device_vector<CollisionPair>(outList.begin(), outList.end());

	thrust::for_each(thrust::device, pairListIn.begin(), pairListIn.end(), GeometricFilterKernel(timestep));
	n = thrust::copy_if(thrust::device, pairListIn.begin(), pairListIn.end(), pairList.begin(), Collision()) - pairList.begin();
	pairList.resize(n);
	outList = thrust::host_vector<CollisionPair>(pairList.begin(), pairList.end());

	for (int i = 0; i < outList.size(); i++) {
		CollisionPair objectPair = outList[i];
		candidateTimeList.clear();
		collisionTimes.clear();

		//TODO - ADD GPU code for time/coplanar filter
		if (objectPair.GetRelativeInclination() == 0)
			candidateTimeList.push_back(-1);
		else
		{
			candidateTimeList = TimeFilter(objectPair, timeStep);
		}

		if (candidateTimeList.size() > 0)
		{
			if (candidateTimeList[0] < 0)
				candidateTimeList = CoplanarFilter(objectPair, timeStep);
		}

		if (candidateTimeList.size() > 0)
		{
			//vector<double> altitudes;
			//collisionTimes = DetermineCollisionTimes(objectPair, candidateTimeList, altitudes);

			pairID = make_pair(objectPair.primaryID, objectPair.secondaryID);
			mass = objectPair.primaryMass + objectPair.secondaryMass;
			for (double candidateTime : candidateTimeList)
			{
				closeTime = CalculateClosestApproachTime(objectPair, candidateTime);
				closeApproach = objectPair.CalculateSeparationAtTime(closeTime);
				collide = closeApproach < (objectPair.GetBoundingRadii() + collisionThreshold);
				if (outputTimes) {

					Event tempEvent(population.GetEpoch() + closeTime, pairID.first, pairID.second, objectPair.GetRelativeVelocity(), mass, objectPair.GetCollisionAltitude(), closeApproach);
					newCollisionTimes.push_back(collide);
					newCollisionList.push_back(tempEvent);
				}
				else if (collide)
				{
					Event tempEvent(population.GetEpoch() + closeTime, pairID.first, pairID.second, objectPair.GetRelativeVelocity(), mass, objectPair.GetCollisionAltitude(), closeApproach);
					newCollisionList.push_back(tempEvent); // Note in this scenario only adds once regardless of number of # potential collisions for pair
					break;
				}
			}
		}
	}
}