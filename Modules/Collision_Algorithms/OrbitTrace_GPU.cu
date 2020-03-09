// OrbitTrace.cpp : contains the implementation of the GPU elements of the Orbit Trace collision algorithm.
//

#include "stdafx.h"
#include "OrbitTrace.h"

// CUDA standard includes
#include <windows.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "device_launch_parameters.h"

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

__host__ list<CollisionPair> CreatePairList_GPU(DebrisPopulation & population)
{
	list<CollisionPair> pairList;
	// TODO - GPU code for creating pairList

	//Talk to Pete about i, j where i < j < N
	return pairList;
}

//TODO - add device function  to operate on each collision pair
__device__ void OrbitTraceAlgorithm(list<CollisionPair>& pairList) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
}

__host__ void OrbitTrace::MainCollision_GPU(DebrisPopulation & population, double timestep)
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
	OrbitTraceAlgorithm <<<numBlocks, numThreads >>> (pairList);
	// 1D iteration over pairList
	//int index = blockIdx.x * blockDim.x + threadIdx.x;

}

