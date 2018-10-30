// Framework.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <fstream>
#include <iostream>
#include <vector>
using namespace std;

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
	int id;

	// Constructor
	DebrisObject()
	{
		std::vector<double> elements(5);
		std::vector<double> anomalies(3);
	}
};