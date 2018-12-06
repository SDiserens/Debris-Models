#include "stdafx.h"
#include "FragmentCloud.h"
#include <vector>

FragmentCloud::FragmentCloud() {}

FragmentCloud::FragmentCloud(double minLength, double maxLength, int buckets=1) : minLength(minLength), maxLength(maxLength)
{
	if (buckets > 1)
		std::vector<FragmentCloud> fragmentBuckets(buckets);
		//FragmentCloud * fragmentBuckets = new FragmentCloud[buckets];
	
	else if (buckets < 1)
		std::invalid_argument(" Recived less than minimum allowed buckets");
	//else
}


FragmentCloud::~FragmentCloud()
{
}


void FragmentCloud::AddFragment(DebrisObject fragment)
{
	fragments.push_back(fragment);
	debrisCount++;
	totalMass += fragment.mass;
}

void FragmentCloud::AddCloud(FragmentCloud debrisCloud)
{
	fragmentBuckets.push_back(debrisCloud);
	debrisCount += debrisCloud.debrisCount;
	totalMass += debrisCloud.totalMass;
}
