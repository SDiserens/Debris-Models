// NASAStdBrkUpMdl.cpp : Contains the implementation of the original NASA Standard Breakup Model.
//

#include "stdafx.h"
#include "NSBM.h"

double FragmentCloud::catastrophicThreshold, FragmentCloud::representativeFragmentThreshold, NSBMFragmentCloud::scaling;
int NSBMFragmentCloud::numFragBuckets, FragmentCloud::representativeFragmentNumber;
string NSBMDebrisFragment::bridgingFunction;

NASABreakupModel::NASABreakupModel()
{
	minLength = 0.001;
	FragmentCloud::catastrophicThreshold           = catastrophicThreshold = 40;
	NSBMFragmentCloud::numFragBuckets                  = numFragBuckets = 30;
	FragmentCloud::representativeFragmentThreshold = representativeFragmentThreshold = 0.02;
	FragmentCloud::representativeFragmentNumber    = representativeFragmentNumber = 10;
	NSBMFragmentCloud::scaling = scaling = 1;
	NSBMDebrisFragment::bridgingFunction = bridgingFunction = "Weighted";
}

NASABreakupModel::NASABreakupModel(double mL) : NASABreakupModel()
{
	minLength = mL;
}

NASABreakupModel::NASABreakupModel(double mL, double cT, int nFB, string bF, double sc, double rFT, int rFN)
{
	minLength = mL;
	catastrophicThreshold = FragmentCloud::catastrophicThreshold = cT;
	numFragBuckets = NSBMFragmentCloud::numFragBuckets = nFB;
	representativeFragmentThreshold = FragmentCloud::representativeFragmentThreshold = rFT;
	representativeFragmentNumber = FragmentCloud::representativeFragmentNumber = rFN;
	bridgingFunction = NSBMDebrisFragment::bridgingFunction = bF;
	NSBMFragmentCloud::scaling = scaling = sc;
}

void  NASABreakupModel::mainBreakup(DebrisPopulation& population, Event& fragmentationEvent)
{
    // Initialise Variables
    bool explosion;

	// Store relevant object variables
	DebrisObject& targetObject = population.GetObject(fragmentationEvent.primaryID);

	if (fragmentationEvent.GetEventType() == 0)
	{
		explosion = true;
		// Simulate primary object breakup
		NSBMFragmentCloud targetDebrisCloud(targetObject, minLength, newSpace);
		MergeFragmentPopulations(population, targetDebrisCloud, fragmentationEvent);
	}

	else
	{
		explosion = false;
		DebrisObject& projectileObject = population.GetObject(fragmentationEvent.secondaryID);
		targetObject.SetTrueAnomaly(fragmentationEvent.primaryAnomaly);
		projectileObject.SetTrueAnomaly(fragmentationEvent.secondaryAnomaly);
		
		// Simulate primary object breakup
		NSBMFragmentCloud targetDebrisCloud(targetObject, projectileObject, minLength, newSpace);
		MergeFragmentPopulations(population, targetDebrisCloud, fragmentationEvent);

		// Simulate secondary object breakup
		NSBMFragmentCloud projectileDebrisCloud(projectileObject, targetObject, minLength, newSpace);
		Event tempEvent(fragmentationEvent);
		tempEvent.SwapPrimarySecondary();
		MergeFragmentPopulations(population, projectileDebrisCloud, tempEvent);
	}

}

void NASABreakupModel::SetNewSpaceParameters()
{
	newSpace = true;
}


NSBMFragmentCloud::NSBMFragmentCloud() // DEfault Constructor
{
}

NSBMFragmentCloud::NSBMFragmentCloud(DebrisObject& targetObject, double init_minLength, bool newS) //Create an explosion Cloud
{
	// Identify key variables
	explosion = true;
	minLength = init_minLength;
	totalMass = targetObject.GetMass();
	maxLength = targetObject.GetLength();
	newSpace = newS;

	// Set parameters for computing fragment distribution
	if (newSpace)
		SetNumberFragmentParametersExplosionNS(targetObject.GetType(), targetObject.GetAreaToMass());
	else
		SetNumberFragmentParametersExplosion();

	// Create Fragment Buckets
	GenerateFragmentBuckets(targetObject);
	targetID = targetObject.GetID();
	projectileID = -1;
}

NSBMFragmentCloud::NSBMFragmentCloud(DebrisObject& targetObject, DebrisObject& projectileObject, double init_minLength, bool newS) //Create a collision Cloud
{
	double collisionKineticEnergy;

	explosion = false;
	minLength = init_minLength;
	// Identify key variables
	totalMass = targetObject.GetMass();
	maxLength = targetObject.GetLength();
	impactMass = projectileObject.GetMass();
	velocity = vector3D(targetObject.GetVelocity());
	relativeVelocity = vector3D(velocity.CalculateRelativeVector(projectileObject.GetVelocity()));
	collisionKineticEnergy = CalculateKineticEnergy(relativeVelocity, projectileObject.GetMass());
	energyMassRatio = CalculateEnergyToMass(collisionKineticEnergy, totalMass);
	newSpace = newS;

	// Set parameters for computing fragment distribution
	if (energyMassRatio > catastrophicThreshold)
		if (newSpace)
			SetNumberFragmentParametersCatastrophicCollisionNS(energyMassRatio);
		else
			SetNumberFragmentParametersCatastrophicCollision();
	else
		SetNumberFragmentParametersCollision();

	// Create Fragment Buckets
	GenerateFragmentBuckets(targetObject);
	targetID = targetObject.GetID();
	projectileID = projectileObject.GetID();
}


//DebrisObject GenerateDebrisObject()
NSBMFragmentCloud::NSBMFragmentCloud(bool init_explosion, double init_minLength, double init_maxLength, int init_numFrag, double init_mass) // Generates a fragment size bucket
{
	explosion = init_explosion;
	maxLength = init_maxLength;
	minLength = init_minLength;
	numFrag = init_numFrag;
	totalMass = init_mass;

	assignedMass = 0;
	debrisCount = 0;
	totalKineticEnergy = totalVolume = 0;
	averageLength = averageSpeed = averageMomentumNorm = 0;

}


void NSBMFragmentCloud::GenerateFragmentBuckets(DebrisObject& targetObject)
{
	double lowerLength, upperLength, logStep, logLength;

	assignedMass = 0;
	SetNumberOfFragments(CalculateNumberOfFragments(minLength));

	// Identify different size buckets for fragments
	lowerLength = minLength;
	logLength = log10(lowerLength);
	logStep = (-logLength) / (numFragBuckets - 1.0);

	// Create set of fragment buckets
	for (int i = 1; i < numFragBuckets; i++)
	{
		// Set upper limits of length
		logLength = logLength + logStep;
		upperLength = pow(10, logLength);

		// Create Bucket
		CreateFragmentBucket(targetObject, lowerLength, upperLength);

		// Update lower limit of length
		lowerLength = upperLength;
	}

	// Create bucket for fragments > 1m
	CreateTopFragmentBucket(targetObject, lowerLength, maxLength);
	UpdateAverageVariables();

	// Check conservation of Mass, Momentum and Energy within limits
	ApplyConservationOfMass();
	ApplyConservationOfMomentum();
	ApplyConservationOfEnergy();
}

void NSBMFragmentCloud::ApplyConservationOfMass()
{
	// Test involved mass vs assigned mass

	// If too little mass
		// Distribute mass as required

	// If too much mass
		// Scale fragments accordingly

}

void NSBMFragmentCloud::ApplyConservationOfMomentum()
{
	double momentumNorm = totalMomentum.vectorNorm();
	if (momentumNorm > NEWTONTOLERANCE)
	{
		double normalisedMomentum = momentumNorm / averageMomentumNorm;
		// Check magnitude of momentum vector compared to average momenmtum
		if (normalisedMomentum < NEWTONTOLERANCE)
			consMomentumFlag = true;
	}
}

void NSBMFragmentCloud::ApplyConservationOfEnergy()
{
	// No way to do this currently as no knowledge about the stored energy involved
}

void NSBMFragmentCloud::CreateTopFragmentBucket(DebrisObject& targetObject, double lowerLength, double upperLength)
{
	int nFrag = -1;
	double remainingMass = totalMass - assignedMass;

	 NSBMFragmentCloud tempFragmentCloud(explosion, lowerLength, upperLength, nFrag, remainingMass);

	if (explosion)
	{
		// Up to 8 fragments
		tempFragmentCloud.numFrag = 7;
		tempFragmentCloud.GenerateDebrisFragments(targetObject);

		energyMassRatio = 0;
		remainingMass -= tempFragmentCloud.assignedMass;
	}

	if (energyMassRatio < catastrophicThreshold)
	{
		// Single large fragment
		tempFragmentCloud.numFrag = 1;
		NSBMDebrisFragment tempFragment(upperLength, remainingMass, explosion, targetObject.GetType());
		tempFragment.SetName(targetObject.GetName() + "-F");

		tempFragment.SetStateVectors(targetObject.GetPosition(), targetObject.GetVelocity() + tempFragment.deltaV);
		//tempFragment.UpdateOrbitalElements(tempFragment.deltaV);
		tempFragment.SetSourceID(targetObject.GetSourceID());
		tempFragment.SetParentID(targetObject.GetID());

		tempFragmentCloud.StoreFragmentVariables(tempFragment);
		tempFragmentCloud.UpdateAverageVariables();

		tempFragmentCloud.fragments.push_back(tempFragment);
	}
	else
	{
		// Assign fragments until mass depleted
		tempFragmentCloud.numFrag = 1;
		while (tempFragmentCloud.assignedMass < remainingMass)
			tempFragmentCloud.GenerateDebrisFragments(targetObject);

	}

	StoreFragmentVariables(tempFragmentCloud);

	// Store fragment bucket
	fragmentBuckets.push_back(tempFragmentCloud);
}

void NSBMFragmentCloud::CreateFragmentBucket(DebrisObject& targetObject, double lowerLength, double upperLength)
{
	double remainingMass = totalMass - assignedMass;

	// Create Fragments
	int nFrag = CalculateBucketFragments(lowerLength, upperLength);
	NSBMFragmentCloud tempFragmentCloud(explosion, lowerLength, upperLength, nFrag, remainingMass);
	
	// Generate debris in each bucket
	tempFragmentCloud.GenerateDebrisFragments(targetObject);

	StoreFragmentVariables(tempFragmentCloud);

	// Store fragment bucket
	fragmentBuckets.push_back(tempFragmentCloud);

}

void NSBMFragmentCloud::GenerateDebrisFragments(DebrisObject& targetObject)
{
	int repFrags, remainingFrags;
	double tempLength;
	double logMaxLength = log10(maxLength);
	double logMinLength = log10(minLength);

	for (int i = 0; i < numFrag; i)
	{
		// Assign fragment length
		tempLength = pow(10, randomNumber(logMinLength, logMaxLength));
	
		// Create new DebrisObject
		// Representative fragment logic
		if (tempLength >= representativeFragmentThreshold)
			repFrags = 1;
		else
		{
			remainingFrags = numFrag - i;
			if (remainingFrags > representativeFragmentNumber)
				repFrags = representativeFragmentNumber;
			else
				repFrags = remainingFrags;
		}
		i += repFrags;

		NSBMDebrisFragment tempFragment(tempLength, explosion, targetObject.GetType(), repFrags);


		if (assignedMass + tempFragment.GetMass() >= totalMass) { 
			// Control maximum mass
			double remainingMass = totalMass - assignedMass;
			tempFragment = NSBMDebrisFragment(tempLength, remainingMass, explosion, targetObject.GetType());
			i = numFrag;
		}

		tempFragment.SetName(targetObject.GetName() + "-F");
		tempFragment.SetSourceID(targetObject.GetSourceID());
		tempFragment.SetParentID(targetObject.GetID());
		// Identify updated orbital elements
		tempFragment.SetStateVectors(targetObject.GetPosition(), targetObject.GetVelocity() + tempFragment.deltaV);
		//tempFragment.UpdateOrbitalElements(tempFragment.deltaV);
		tempFragment.SetInitEpoch(NAN);

		// Update FragmentCloud variables for bucket
		StoreFragmentVariables(tempFragment);

		// Add temp fragment to fragments vector
		fragments.push_back(tempFragment);

	}

	UpdateAverageVariables();
}

void NSBMFragmentCloud::StoreFragmentVariables(NSBMFragmentCloud& tempFragmentCloud)
{
	int nFrag = tempFragmentCloud.debrisCount;
	// Update overall FragmentCloud variables
	debrisCount += nFrag;
	assignedMass += tempFragmentCloud.assignedMass;
	totalKineticEnergy += tempFragmentCloud.totalKineticEnergy;
	totalVolume += tempFragmentCloud.totalVolume;

	averageLength += tempFragmentCloud.averageLength * nFrag;
	averageSpeed += tempFragmentCloud.averageSpeed * nFrag;
	averageMomentumNorm += tempFragmentCloud.averageMomentumNorm * nFrag;

	averageVelocity = averageVelocity + tempFragmentCloud.averageVelocity * nFrag;
	totalMomentum = totalMomentum + tempFragmentCloud.totalMomentum;
}


void NSBMFragmentCloud::StoreFragmentVariables(NSBMDebrisFragment& tempFragment)
{
	// Update FragmentCloud variables for bucket
	int nFrag = tempFragment.GetNFrag();
	debrisCount += nFrag;
	assignedMass += tempFragment.GetMass() * nFrag;
	totalKineticEnergy += tempFragment.kineticEnergy * nFrag;
	totalVolume += tempFragment.volume * nFrag;

	averageLength += tempFragment.GetLength() * nFrag;
	averageSpeed += tempFragment.deltaVNorm * nFrag;
	averageMomentumNorm += tempFragment.deltaVNorm * tempFragment.GetMass() * nFrag;

	averageVelocity = averageVelocity + tempFragment.deltaV * nFrag;
	totalMomentum = totalMomentum + tempFragment.deltaV * tempFragment.GetMass() * nFrag;
}


void NSBMFragmentCloud::UpdateAverageVariables()
{
	double ratio;
	if (debrisCount == 0)
		ratio = 0.0;
	else
		ratio = 1.0 / debrisCount;

	averageMass = assignedMass * ratio;
	averageKineticEnergy = totalKineticEnergy * ratio;
	averageVolume = totalVolume * ratio;

	averageLength *= ratio;
	averageSpeed *= ratio;
	averageMomentumNorm *= ratio;
	averageDensity = averageMass / averageVolume;

	averageVelocity = averageVelocity * ratio;
	averageMomentum = totalMomentum * ratio;
}



int NSBMFragmentCloud::CalculateNumberOfFragments(double length)
{
	int nFrag = (int)round(nFragCoefficient * pow(length, nFragExponent));
	return nFrag;
}

int NSBMFragmentCloud::CalculateBucketFragments(double lowerLength, double upperLength)
{
	int nFrag = CalculateNumberOfFragments(lowerLength) - CalculateNumberOfFragments(upperLength);
	return nFrag;
}

void NSBMFragmentCloud::SetNumberFragmentParametersExplosion()
{
	nFragExponent = -1.6;
	nFragCoefficient = scaling * 6;
}

void NSBMFragmentCloud::SetNumberFragmentParametersCollision()
{
	double ejectaMass = impactMass * relativeVelocity.vectorNorm2();
	nFragCoefficient = 0.1 * pow(ejectaMass, 0.75);
	nFragExponent = -1.71;
}

void NSBMFragmentCloud::SetNumberFragmentParametersCatastrophicCollision()
{
	double ejectaMass = totalMass;
	nFragCoefficient = 0.1 * pow(ejectaMass, 0.75);
	nFragExponent = -1.71;
}

void NSBMFragmentCloud::SetNumberFragmentParametersExplosionNS(bool type, double a2m)
{
	if (type == 1) {
		nFragExponent = -2 - 0.1/sqrt(a2m);
		nFragCoefficient = 0.1 * totalMass/800;
	}

	else {
		nFragExponent = -1.6;
		nFragCoefficient = scaling * 6;
	}
}

void NSBMFragmentCloud::SetNumberFragmentParametersCatastrophicCollisionNS(double eMr)
{
	double ejectaMass = totalMass;
	nFragCoefficient = 0.02 * pow(ejectaMass, 0.75);
	nFragExponent = -2.2 - 100/sqrt(eMr);
}

void NSBMFragmentCloud::SetNumberOfFragments(int nFrag)
{
	numFrag = nFrag;
}


// Debris Fragments
NSBMDebrisFragment::NSBMDebrisFragment(double init_length, bool init_explosion, int source, int numFrag)
{
	objectID = ++objectSEQ;
	sourceType = source;
	objectType = 2;
	nFrag = numFrag;
	//std::default_random_engine generator;
	length = init_length;
	radius = length / 2.0;
	lambda = log10(length);
	explosion = init_explosion;
	CalculateArea();
	isIntact = false;
	isActive = false;
	explosionProbability = 0.;

	if (explosion)
		sourceEvent = 1;
	else
		sourceEvent = 2;

	//Small Fragments
	if (length < 0.08)
	{
		SetSmallAreaMassParameters();
	}
	//Bridged Fragments
	else if (length < 0.11)
	{
		AreaMassBridgingFunction();
	}
	// UpperStage Fragments
	else if (sourceType == 0)
	{
		SetUpperStageAreaMassParameters();
	}
	// Spacecraft Fragments
	else
	{
		SetSpacecraftAreaMassParameters();
	}

	GenerateAreaToMassValue();

	CalculateMassFromArea();
	CalculateRelativeVelocity();
	kineticEnergy = CalculateKineticEnergy(deltaVNorm, GetMass());
	CalculateVolume();
}

NSBMDebrisFragment::NSBMDebrisFragment(double tempLength, double init_mass, bool init_explosion, int source, int numFrag)
{
	objectID = ++objectSEQ;
	lambda = log10(tempLength);
	mass = init_mass;
	nFrag = numFrag;
	explosion = init_explosion;
	CalculateArea();
	isIntact = false;
	isActive = false;
	explosionProbability = 0.;

	if (explosion)
		sourceEvent = 1;
	else
		sourceEvent = 2;

	objectType = 2;
	sourceType = source;

	// UpperStage Fragments
	if (sourceType == 0)
	{
		SetUpperStageAreaMassParameters();
	}
	// Spacecraft Fragments
	else
	{
		SetSpacecraftAreaMassParameters();
	}

	GenerateAreaToMassValue();
	CalculateAreaFromMass();
	CalculateLength();
	radius = length / 2.0;
	CalculateAreaToMass();
	chi = log10(areaToMass);
	CalculateRelativeVelocity();
	kineticEnergy = CalculateKineticEnergy(deltaVNorm, GetMass());
	CalculateVolume();
}


void NSBMDebrisFragment::SetUpperStageAreaMassParameters()
{
	// alpha
	if (lambda <= -1.4)
		alpha = 1;
	else if (lambda >= 0)
		alpha = 0.5;
	else
		alpha = 1 - 0.3571 * (lambda + 1.4);

	// mu_1
	if (lambda <= -0.5)
		mu_1 = -0.45;
	else if (lambda >= 0)
		mu_1 = -0.9;
	else
		mu_1 = -0.45 - 0.9 * (lambda + 0.5);

	// sigma_1
	sigma_1 = 0.55;

	// mu_2
	mu_2 = -0.9;

	// sigma_2
	if (lambda <= -1.0)
		sigma_2 = 0.28;
	else if (lambda >= 0.1)
		sigma_2 = 0.1;
	else
		sigma_2 = 0.28 - 0.1636 * (lambda + 1);
}

void NSBMDebrisFragment::SetSpacecraftAreaMassParameters()
{
	// alpha
	if (lambda <= -1.95)
		alpha = 0;
	else if (lambda >= 0.55)
		alpha = 1.0;
	else
		alpha = 0.3 + 0.4 * (lambda + 1.2);

		// mu_1
	if (lambda <= -1.1)
		mu_1 = -0.6;
	else if (lambda >= 0)
		mu_1 = -0.95;
	else
		mu_1 = -0.6 - 0.318 * (lambda + 1.1);

		// sigma_1
	if (lambda <= -1.3)
		sigma_1 = 0.1;
	else if (lambda >= -0.3)
		sigma_1 = 0.3;
	else
		sigma_1 = 0.1 + 0.2 * (lambda + 1.3);

		// mu_2
	if (lambda <= -0.7)
		mu_2 = -1.2;
	else if (lambda >= -0.1)
		mu_2 = -2.0;
	else
		mu_2 = -1.2 - 1.333 * (lambda + 0.7);

		// sigma_2
	if (lambda <= -0.5)
		sigma_2 = 0.5;
	else if (lambda >= -0.3)
		sigma_2 = 0.3;
	else
		sigma_2 = 0.5 - 1 * (lambda + 0.5);
}

void NSBMDebrisFragment::AreaMassBridgingFunction()
{
	// Bridging function
	//string bridgingFunction = "Small";
	//string bridgingFunction = "Large";
	//string bridgingFunction = "Mean";
	//string bridgingFunction = "Weighted";

	SetSmallAreaMassParameters();

	if (bridgingFunction != "Small")
	{
		// Store small parameters
		double alphaSmall, mu_1Small, sigma_1Small, mu_2Small, sigma_2Small;
		alphaSmall = alpha;
		mu_1Small = mu_1;
		sigma_1Small = sigma_1;
		mu_2Small = mu_2;
		sigma_2Small = sigma_2;

		// Generate large parameters
		// UpperStage Fragments
		if (sourceType == 0)
			SetUpperStageAreaMassParameters();
		// Spacecraft Fragments
		else
			SetSpacecraftAreaMassParameters();

		if (bridgingFunction == "Mean")
		{	
			// Average
			alpha   = (alphaSmall   + alpha)   / 2;
			mu_1    = (mu_1Small    + mu_1)    / 2;
			sigma_1 = (sigma_1Small + sigma_1) / 2;
			mu_2    = (mu_2Small    + mu_2)    / 2;
			sigma_2 = (sigma_2Small + sigma_2) / 2;
		}

		else if (bridgingFunction == "Weighted")
		{
			// Weighted Average
			double weight = (length - 0.08) / (0.03);
			double smallWeight = 1 - weight;
			alpha   = (smallWeight * alphaSmall   + weight * alpha);
			mu_1    = (smallWeight * mu_1Small    + weight * mu_1);
			sigma_1 = (smallWeight * sigma_1Small + weight * sigma_1);
			mu_2    = (smallWeight * mu_2Small    + weight * mu_2);
			sigma_2 = (smallWeight * sigma_2Small + weight * sigma_2);
		}
	}
}

void NSBMDebrisFragment::SetSmallAreaMassParameters()
{
	objectType = 2;

	alpha = 1;

	if (lambda <= -1.75)
		mu_1 = -0.3;
	else if (lambda >= -1.25)
		mu_1 = -1.0;
	else
		mu_1 = -0.3 - 1.4 * (lambda + 1.75);

	if (lambda <= -3.5)
		sigma_1 = 0.2;
	else
		sigma_1 = 0.2 + 0.1333 * (lambda + 3.5);

	mu_2 = 0.0;
	sigma_2 = 1.0;
}

void NSBMDebrisFragment::GenerateAreaToMassValue()
{
	std::normal_distribution<double> distribution1(mu_1, sigma_1);
	std::normal_distribution<double> distribution2(mu_2, sigma_2);
	chi = alpha * distribution1(*generator) + (1 - alpha) * distribution2(*generator);
	areaToMass = pow(10, chi);
}


void NSBMDebrisFragment::CalculateArea()
{
	if (length < 0.00167)
		area = 0.540424 * length * length;
	else
		area = 0.556945 * pow(length, 2.0047077);
}

void NSBMDebrisFragment::CalculateLength()
{
	length = pow(area / 0.556945, 1.0/ 2.0047077);
}

void NSBMDebrisFragment::CalculateRelativeVelocity()
{
	double nu, mu, sigma, theta, phi;

	sigma = 0.4;
	if (explosion)
		mu = 0.2 * chi + 1.85;
	else
		mu = 0.9 * chi + 2.9;

	std::normal_distribution<double> velocityDistribution(mu, sigma);

	nu = velocityDistribution(*generator);

	deltaVNorm = 0.001 * pow(10, nu);
	theta = randomNumberPi();
	phi = randomNumberTau();

	deltaV = vector3D(deltaVNorm * sin(theta) * cos(phi),
					  deltaVNorm * sin(theta) * sin(phi),
					  deltaVNorm * cos(theta)			);
}

void NSBMDebrisFragment::CalculateVolume()
{
	volume = length * length * length;
	density = mass / density;
}

