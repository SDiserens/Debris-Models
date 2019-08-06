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
	bool collisionDetail;
	double fragmentMinLength, collisionThreshold;

	//TODO - Create independent function for config 
	Json::Value config;
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

	breakUpType = config["Fragmentation"].asString();
	fragmentMinLength = config["minLength"].asDouble();

	collisionType = config["CollsionAlgorithm"].asString();
	collisionDetail = config["CollisionDetail"].asBool();
	collisionThreshold = config["CollisionThreshold1"].asDouble();

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
	LoadScenario(environmentPopulation, populationFilename);
	
	// Load Modules
	auto propagator = ModuleFactory::CreatePropagator(propagatorType, environmentPopulation);

	auto collisionModel = ModuleFactory::CreateCollisionAlgorithm(collisionType, collisionDetail, collisionThreshold);
	// TODO - Include configuration variable fro breakup model
	auto breakUp = ModuleFactory::CreateBreakupModel(breakUpType, fragmentMinLength);

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


