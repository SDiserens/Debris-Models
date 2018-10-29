// Framework.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <fstream>
#include <iostream>
using namespace std;

int main()
{
	// ----------------------------
	// ------ Initialisation ------
	// ----------------------------
	// Read config file

	// Initialise population
	InitPopulation()

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

void InitPopulation(char *populationFilename)
{
	// Initialise variables
	ifstream popFile;

	// Open population file
	popFile.open(populationFilename);

	// Read population data object by object
		// Create object

		// Add object to population

	// Close file
	popfile.close();
}

class DebrisObject
{
	public:
	int id;

	// Constructor
	DebrisObject()
	{
		std::vector<double> elements;
	}
};