#include "stdafx.h"
#include "Analytic_J2.h"


Analytic_J2::Analytic_J2(bool useJ2=true)
{
	useJ2 = useJ2;
}


Analytic_J2::~Analytic_J2()
{
}

void Analytic_J2::UpdateElements(DebrisObject &object, double timeStep)
{
	double deltaM, deltaSMA=0, deltaEcc = 0, deltaInc = 0, deltaRA = 0, deltaAP = 0;
	double period, revolutions, gravityTerm;
	OrbitalElements& initialElements = object.GetElements();

	period = initialElements.CalculatePeriod();
	revolutions = timeStep / period;

	if (useJ2)
	{
		gravityTerm = 3 * Pi * J2term * radiusEarth * radiusEarth / (initialElements.semiMajorAxis * initialElements.semiMajorAxis * pow((1. - initialElements.eccentricity*initialElements.eccentricity),2.));
	
		deltaRA = -1 * revolutions * gravityTerm * cos(initialElements.inclination);
		deltaAP = revolutions * gravityTerm * (2 - 2.5 * sin(initialElements.inclination) * sin(initialElements.inclination));
	}

	initialElements.UpdateOrbitalElements(deltaRA = deltaRA, deltaAP = deltaAP);

	deltaM = Tau * revolutions;
	
	initialElements.SetMeanAnomaly(TauRange(initialElements.GetMeanAnomaly() + deltaM));
}
