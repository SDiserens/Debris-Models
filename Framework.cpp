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
	int mcRuns;

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
	mcRuns = config["MonteCarlo"].asInt();

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

	// Load population
	auto& propagator = ModuleFactory::CreatePropagator(propagatorType, environmentPopulation, propagatorConfig);
	// Load Modulea
	auto& collisionModel = ModuleFactory::CreateCollisionAlgorithm(collisionType, collisionConfig);

	auto& breakUp = *ModuleFactory::CreateBreakupModel(breakUpType, fragmentationConfig);

	// Load Environment Parameters
	elapsedDays = 0;
	simulationDays = config["Duration"].asDouble();
	stepDays = config["StepSize"].asDouble();

	vector<pair<long, long>> collisionList;
	vector<tuple<int, double, pair<string, string>, double>> collisionLog;

	for (int i = 0; i < mcRuns; i++)
	{
		LoadScenario(environmentPopulation, populationFilename);
		propagator->SyncPopulation();

		// Validate Modules


		// --------------------------
		// --- Evolve Environment ---
		// --------------------------

		elapsedDays = 0;
		// While timeSimulation < timeEnd
		while (elapsedDays < simulationDays)
		{
			// Propagate Object Orbits
			timeStep = min(min(stepDays, environmentPopulation.GetTimeToNextInitEpoch()), simulationDays - elapsedDays);
			(*propagator).PropagatePopulation(timeStep);
			elapsedDays += timeStep;

			// Determine Events
				// Collision Detection
			collisionModel->MainCollision(environmentPopulation, timeStep * secondsDay);
			collisionList = collisionModel->GetNewCollisionList();

			// if extra output requested
			if (collisionConfig["Verbose"].asBool()) {
				vector<double> collisionOutput;
				// Retrieve collision output
				collisionOutput = collisionModel->GetNewCollisionVerbose();

				// Log data
				for (int i = 0; i < collisionList.size(); i++) {
					// TODO - Use Pair ID values to retrieve object names/noradID for greater clarity

					collisionLog.push_back(make_tuple(i, elapsedDays, make_pair(environmentPopulation.GetObject(collisionList[i].first).GetName(), 
																				environmentPopulation.GetObject(collisionList[i].second).GetName()), collisionOutput[i]));
				}
			}

			// For each pair in collision list
				// determine if collision avoidance occurs

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

		//TODO - Save Event log for MC run
		}
	}
	// ----------------------------
	// ------ End Simulation ------
	// ----------------------------
	// Save final population

	// Write Logs to output files
	if (collisionConfig["Verbose"].asBool()) {

		// TODO - Pass population data
		WriteCollisionData(populationFilename, config, collisionType, collisionConfig, collisionLog);
	}

    return 0;
}


