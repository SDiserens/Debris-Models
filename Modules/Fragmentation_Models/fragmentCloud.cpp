

#include "stdafx.h"
#include "FragmentCloud.h"


FragmentCloud::FragmentCloud(double minLength, double maxLength) : minLength(minLength), maxLength(maxLength)
{
	
}


FragmentCloud::~FragmentCloud()
{
}


void FragmentCloud::AddFragment(DebrisObject fragment)
{
	debrisCount++;
}

void FragmentCloud::AddCloud(FragmentCloud fragmentCloud)
{
	debrisCount += fragmentCloud.debrisCount;
}
