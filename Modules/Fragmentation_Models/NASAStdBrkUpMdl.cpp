// NASAStdBrkUpMdl.cpp : Contains the implementation of the original NASA Standard Breakup Model.
//

#include "../../Framework/stdafx.h"
#include "fragmentCloud.h"

class fragmentCloud;

fragmentCloud generateExplosion(), generateCollision();


fragmentCloud mainBreakup(DebrisObject& TargetObject, DebrisObject *projectilePointer=NULL, float minLength=0.001)
{
    // Initialise Variables
    bool explosion;
	float maxLength = TargetObject.length;
	fragmentCloud debrisCloud;
    
	if (projectilePointer == NULL)
	{
		explosion = true;
		debrisCloud = generateExplosion();
	}

	else
	{
		explosion = false;
		DebrisObject& ProjectileObject = *projectilePointer;
		delete projectilePointer;
		debrisCloud = generateCollision();
	}

	return debrisCloud;
}

fragmentCloud generateExplosion()
{

}

fragmentCloud generateCollision()
{

}