#pragma once
class vector3D
{
public:
	double x, y, z;

public:
	vector3D();
	vector3D(double X, double Y, double Z);
	double vectorNorm(int ord);
	double vectorNorm2();
	vector3D CalculateRelativeVector(vector3D vectorB);
	void addVector(vector3D vectorB);
};

double CalculateKineticEnergy(vector3D relativeVelocity, double mass);
