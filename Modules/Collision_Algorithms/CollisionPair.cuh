
#include "MOID\distlink.h"
#include "MOID\MOID.h"

class CollisionPair
{
public:
	OrbitalElements primaryElements, secondaryElements;
	long primaryID, secondaryID;
	double approachAnomalyP, approachAnomalyS, probability=0, minSeperation, primaryAnomaly, secondaryAnomaly, primaryMass, secondaryMass;
	bool coplanar, collision;
	int overlapCount;
	double relativeInclination, boundingRadii;
protected:
	double relativeVelocity, deltaPrimary, deltaSecondary, deltaPrimary2, deltaSecondary2, collisionAltitude;

public:
	CUDA_CALLABLE_MEMBER CollisionPair();
	CUDA_CALLABLE_MEMBER CollisionPair(DebrisObject& objectI, DebrisObject& objectJ);
	CUDA_CALLABLE_MEMBER CollisionPair(long IDI, long IDJ);
	CUDA_CALLABLE_MEMBER double GetRelativeInclination();
	CUDA_CALLABLE_MEMBER void SetCollisionPair(DebrisObject objectI, DebrisObject objectJ);
	vector<double>  CalculateAngularWindowPrimary(double distance);
	vector<double>  CalculateAngularWindowSecondary(double distance);
	vector3D GetPrimaryPositionAtTime(double timeFromEpoch);
	vector3D GetPrimaryVelocityAtTime(double timeFromEpoch);
	vector3D GetSecondaryPositionAtTime(double timeFromEpoch);
	vector3D GetSecondaryVelocityAtTime(double timeFromEpoch);
	void GenerateArgumenstOfIntersection();
	double CalculateSeparationAtTime(double timeFromEpoch);
	CUDA_CALLABLE_MEMBER double CalculateMinimumSeparation();
	CUDA_HOST_MEMBER double CalculateMinimumSeparation_DL();
	CUDA_HOST_MEMBER double CalculateMinimumSeparation_MOID();
	CUDA_CALLABLE_MEMBER void CalculateArgumenstOfIntersection();
	CUDA_CALLABLE_MEMBER void CalculateArgumenstOfIntersectionCoplanar();
	CUDA_CALLABLE_MEMBER void CalculateRelativeInclination();
	CUDA_CALLABLE_MEMBER double GetBoundingRadii();
	CUDA_CALLABLE_MEMBER double GetCollisionAltitude();
	CUDA_CALLABLE_MEMBER void SetCollisionAltitude(double altitude);
	CUDA_CALLABLE_MEMBER void SetRelativeVelocity(double relV);
	CUDA_CALLABLE_MEMBER double GetRelativeVelocity();

protected:
	vector<double> CalculateAngularWindow(OrbitalElements& elements, double distance, double delta);
};