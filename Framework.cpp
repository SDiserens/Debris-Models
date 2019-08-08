// Framework.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
using namespace std;

int main(int argc, char** argv)
{
	// ----------------------------
	// ------ Initialisation ------
	// ----------------------------
	string arg, populationFilename, propagatorType, breakUpType, collisionType;
	double timeStep, stepDays, elapsedDays, simulationDays;

	//TODO - Create independent function for config 
	Json::Value config, propagatorConfig, fragmentationConfig, collisionConfig;
	Json::Reader reader;
	
	cout << "Reading Config File...";
	// Read config file
	ifstream configFile("config.json");
	// Parse config file to identify scenario file and settings
	reader.parse(configFile, config);
	cout << " Parsing Config...";

	// Parsing config variables
	populationFilename = config["scenarioFilename"].asString();

	propagatorType = config["Propagator"].asString();
	propagatorConfig = config["PropagatorConfig"];

	breakUpType = config["Fragmentation"].asString();
	fragmentationConfig = config["FragmentationConfig"];

	collisionType = config["CollsionAlgorithm"].asString();
	collisionConfig = config["CollisionConfig"];

	// Parse command line arguments
	for (int i = 1; i < argc; ++i) {
		arg = argv[i];
		if ((arg == "-f") || (arg == "--filename"))
		{
			populationFilename = argv[++i];
		}
		if ((arg == "-h") || (arg == "--help"))
		{
			//TODO - Create help output
		}
	}
	// Initialise population
	DebrisPopulation environmentPopulation;

	// Load Modules
	auto& propagator = *ModuleFactory::CreatePropagator(propagatorType, environmentPopulation, propagatorConfig);

	auto& collisionModel = *ModuleFactory::CreateCollisionAlgorithm(collisionType, collisionConfig);

	auto& breakUp = *ModuleFactory::CreateBreakupModel(breakUpType, fragmentationConfig);

	// Load population
	LoadScenario(environmentPopulation, populationFilename);
	propagator.SyncPopulation();

		// Validate Modules
	
	// Load Environment Parameters
	elapsedDays = 0;
	simulationDays = config["Duration"].asDouble();
	stepDays = config["StepSize"].asDouble();

	// --------------------------
	// --- Evolve Environment ---
	// --------------------------
	// While timeSimulation < timeEnd
	while (elapsedDays < simulationDays)
	{
		// Propagate Object Orbits
		timeStep = secondsDay * min(stepDays, environmentPopulation.GetNextInitEpoch(), simulationDays - elapsedDays);
		propagator.PropagatePopulation(timeStep);
		elapsedDays += timeStep;

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
	}

	// ----------------------------
	// ------ End Simulation ------
	// ----------------------------
	// Save final population

	// Write Logs to output files

    return 0;
}


