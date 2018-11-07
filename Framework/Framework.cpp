// Framework.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <fstream>
#include <iostream>
#include <vector>
using namespace std;

int objectSEQ = 0;
void InitPopulation(string populationFilename);

int main()
{
	// ----------------------------
	// ------ Initialisation ------
	// ----------------------------
	// Read config file
	string populationFilename;

	// Initialise population
	InitPopulation(populationFilename);

	// Load Modules

		// Validate Modules

	// Load Environment Parameters

	// --------------------------
	// --- Evolve Environment ---
	// --------------------------
	// While timeSimulation < timeEnd
		// Propagate Object Orbits

		// Determine Events
			// Collision Detection

				// Log

			// Generate Explosions

				// Log

			// Check for Pre-specified Events

				// Log

		// Update population
			// Simulate Fragmentations

				// Log

			// Generate Launches

				// Log

			// Remove Decayed Objects

				// Log


	// ----------------------------
	// ------ End Simulation ------
	// ----------------------------
	// Save final population

	// Write Logs to output files

    return 0;
}

void InitPopulation(string populationFilename)
{
	// Initialise variables
	ifstream popFile;

	// Open population file
	popFile.open(populationFilename.c_str());

	// Read population data object by object
		// Create object

		// Add object to population

	// Close file
	popFile.close();
}

class DebrisObject
{
public:
	long objectID;

private:
	std::vector<double> elements(5); // semi-major axis, eccentricity, inclination, right ascension of ascending node, arguement of perigee
	std::vector<double> anomalies(3); // mean anomaly, eccentric anomaly, true anomaly
	float radius, mass, length, meanAnomalyEpoch;
	long parentID, sourceID;

	// Constructor
	DebrisObject(float init_radius, float init_mass, float init_length, double semiMajorAxis, double eccentricity, double inclination,
				 double rightAscension, double argPerigee, double init_meanAnomaly)
	{
		objectID = ++objectSEQ; // This should be generating a unique ID per object incl. when threading
		radius = init_radius;
		mass = init_mass;
		length = init_length;
		elements[0] = semiMajorAxis;
		elements[1] = eccentricity;
		elements[2] = inclination;
		elements[3] = rightAscension;
		elements[4] = argPerigee;
		meanAnomalyEpoch = anomalies[0] = init_meanAnomaly;
	}
};