// Framework.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Modules\Modules.h"
#include "Utilities\ModuleFactory.h"
using namespace std;

int main(int argc, char** argv)
{
	// ----------------------------
	// ------ Initialisation ------
	// ----------------------------
	string arg, populationFilename, propagatorType, breakUpType, collisionType, ouputName;
	double timeStep, stepDays, elapsedDays, simulationDays, threshold, avoidanceProbability=0;
	bool setThreshold = false;
	int mcRuns;

	//TODO - Create independent function for config 
	Json::Value config, propagatorConfig, fragmentationConfig, collisionConfig;
	Json::Reader reader;
	
	cout << "Reading Config File...\n";
	// Read config file
	ifstream configFile("config.json");
	
	// Parse config file to identify scenario file and settings
	cout << " Parsing Config...\n";
	reader.parse(configFile, config);
	

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
			/*
			-b or --fragmentation
			"Fragmentation": null, // "NSBM",

			-p or --propagator
			"Propagator": "SGP4", // "SGP4", "SimpleJ2"

			-c or --collision
			"CollsionAlgorithm": "OrbitTrace", // "Cube", "OrbitTrace", "Hoots"
			*/
		}
		if ((arg == "-c") || (arg == "--collision"))
		{
			collisionType = argv[++i];
		}
		if ((arg == "-mc") || (arg == "--montecarlo"))
		{
			mcRuns = stoi(argv[++i]);
		}
		if ((arg == "-b") || (arg == "--fragmentation"))
		{
			breakUpType = argv[++i];
		}
		if ((arg == "-p") || (arg == "--propagator"))
		{
			propagatorType = argv[++i];
		}
		if ((arg == "-t") || (arg == "--threshold"))
		{
			setThreshold = true;
			threshold = stod(argv[++i]);
		}
	}

	// Initialise population
	DebrisPopulation initPopulation, environmentPopulation;

	// Load population
	auto& propagator = ModuleFactory::CreatePropagator(propagatorType, environmentPopulation, propagatorConfig);
	// Load Modules
	auto& collisionModel = ModuleFactory::CreateCollisionAlgorithm(collisionType, collisionConfig);
	if (setThreshold)
		collisionModel->SetThreshold(threshold);

	auto& breakUp = *ModuleFactory::CreateBreakupModel(breakUpType, fragmentationConfig);

	// Load Environment Parameters
	elapsedDays = 0;
	stepDays = config["StepSize"].asDouble();

	vector<tuple<int, double, int, tuple<int, int, int>, int, tuple<int, int, int>>> simulationLog; // (MC, #days, #objects, (upperstage, spacecraft, debris), #events, (Explosion, Collision, Collision Avoidance)) 
																									
	vector<Event> collisionList;
	vector<tuple<int, double, pair<string, string>, double, double>> collisionLog; // (MC, #days, objectIDs, probability, altitude)
	vector<double> collisionOutput;
	DebrisObject target;
	DebrisObject projectile;

	cout << "Reading Population File : " + populationFilename + "...\n";
	LoadScenario(initPopulation, populationFilename);
	populationFilename = populationFilename.substr(0, populationFilename.find("."));
	simulationDays = initPopulation.GetDuration();

	cout << "Running " + to_string(mcRuns) + " simulations of " + to_string(simulationDays) + " days, using " + propagatorType + ", " + breakUpType + " and " + collisionType + "...\n";
	for (int j = 0; j < mcRuns; j++)
	{
		environmentPopulation = DebrisPopulation(initPopulation);
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
			// TODO - identify relative velocity at collision point
			// TODO add altitude logging
			collisionList = collisionModel->GetNewCollisionList();

			// if extra output requested
			if (collisionConfig["Verbose"].asBool()) {

				// Retrieve collision output
				collisionOutput = collisionModel->GetNewCollisionVerbose();

				// Log data
				for (int i = 0; i < collisionList.size(); i++) {

					collisionLog.push_back(make_tuple(j, elapsedDays, make_pair(to_string(environmentPopulation.GetObject(collisionList[i].primaryID).GetNoradID()), 
																				to_string(environmentPopulation.GetObject(collisionList[i].secondaryID).GetNoradID())),
											collisionOutput[i], collisionList[i].altitude));
				}
				collisionOutput.clear();
			}

			// For each pair in collision list
			for (auto collision : collisionList) {
				// determine if collision avoidance occurs
				target = environmentPopulation.GetObject(collision.primaryID);
				projectile = environmentPopulation.GetObject(collision.secondaryID);
				avoidanceProbability = 1 - (1 - target.GetAvoidanceSuccess()) * (1 - projectile.GetAvoidanceSuccess());
				if (collisionModel->DetermineCollisionAvoidance(avoidanceProbability)) {
					// Log
					collision.CollisionAvoidance();
					environmentPopulation.AddDebrisEvent(collision);
				}
				else {
				// Simulate Fragmentations
					//TODO - ADD fragmentation
				}
			}
			// Generate Explosions

			// Check for Pre-specified Events

			// Generate Launches

				// Log

			// Remove Decayed Objects

				// Log

			simulationLog.push_back(tuple_cat(make_tuple(j), environmentPopulation.GetPopulationState()));
				
		}

		// ----------------------------
		// ------ End Simulation ------
		// ----------------------------
		// Save final population
		ouputName = populationFilename + "_#" + to_string(j + 1);
		// Write Logs to output files
		if (collisionConfig["Verbose"].asBool()) {

			// TODO - Fix issue with cmd line threshold setting
			WriteCollisionData(ouputName, config, collisionType, collisionConfig, collisionLog);
			collisionLog.clear();
		}
		WriteEventData(ouputName, config, collisionType, collisionConfig, propagatorType, propagatorConfig, breakUpType, fragmentationConfig, environmentPopulation.eventLog);
		WriteSimulationData(ouputName, config, collisionType, collisionConfig, propagatorType, propagatorConfig, breakUpType, fragmentationConfig, simulationLog);
		simulationLog.clear();
	}

    return 0;
}


