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

	// Config objects
	Json::Value config, propagatorConfig, fragmentationConfig, collisionConfig;

	// Variable
	string arg, populationFilename, propagatorType, breakUpType, collisionType, ouputName;
	double timeStep, stepDays, elapsedDays, simulationDays, threshold, avoidanceProbability=0;
	bool  logging = true, setThreshold = false;
	int mcRuns, i;
	volatile int n;
	DebrisObject target, projectile;

	// Data logs
	vector<tuple<int, double, int, tuple<int, int, int>, int, tuple<int, int, int>>> simulationLog;
		// (MC, #days, #objects, (upperstage, spacecraft, debris), #events, (Explosion, Collision, Collision Avoidance)) 
	vector<tuple<int, double, pair<string, string>, double, double>> collisionLog;
		// (MC, #days, objectIDs, probability, altitude)
	vector<Event> collisionList;
	vector<Event> explosionList;
	vector<Event> definedList;
	vector<double> collisionOutput;
	
	// ----------------------------
	// - Parsing config variables -
	// ----------------------------
	LoadConfigFile(config);

	populationFilename = config["scenarioFilename"].asString();
	mcRuns = config["MonteCarlo"].asInt();
	logging = config["logging"].asBool();

	propagatorType = config["Propagator"].asString();
	propagatorConfig = config["PropagatorConfig"];

	breakUpType = config["Fragmentation"].asString();
	fragmentationConfig = config["FragmentationConfig"];

	collisionType = config["CollsionAlgorithm"].asString();
	collisionConfig = config["CollisionConfig"];
	int moid = 0;

	// ----------------------------
	// Parse command line arguments
	// ----------------------------
	for (i = 1; i < argc; ++i) {
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
		if ((arg == "-l") || (arg == "--logging"))
		{
			logging = stod(argv[++i]);
		}
	}

	// ----------------------------
	// --- Initialise population --
	// ----------------------------
	DebrisPopulation initPopulation, environmentPopulation;

	// Load population
	auto& propagator = ModuleFactory::CreatePropagator(propagatorType, environmentPopulation, propagatorConfig);

	// ----------------------------
	// ------- Load Modules -------
	// ----------------------------
	auto& collisionModel = ModuleFactory::CreateCollisionAlgorithm(collisionType, collisionConfig);
	if (collisionConfig["ParallelGPU"].asBool())
		collisionModel->SwitchParallelGPU();
	if (collisionConfig["ParallelCPU"].asBool())
		collisionModel->SwitchParallelCPU();
	if (setThreshold) {
		collisionModel->SetThreshold(threshold);
		ModuleFactory::UpdateCollisionThreshold(collisionType, collisionConfig, threshold);
	}

	auto& breakUp = ModuleFactory::CreateBreakupModel(breakUpType, fragmentationConfig);

	// ----------------------------
	// Load Environment Parameters
	// ----------------------------
	elapsedDays = 0;
	stepDays = config["StepSize"].asDouble();

	cout << "Reading Population File : " + populationFilename + "...\n";
	LoadScenario(initPopulation, populationFilename);
	populationFilename = populationFilename.substr(0, populationFilename.find("."));
	simulationDays = initPopulation.GetDuration();

	// ----------------------------
	//  Simulate Environment Runs  
	// ----------------------------
	cout << "Running " + to_string(mcRuns) + " simulations of " + to_string(simulationDays) + " days, using " + propagatorType + ", " + breakUpType + " and " + collisionType + "...\n";
	for (int j = 0; j < mcRuns; j++)
	{
		environmentPopulation = DebrisPopulation(initPopulation);
		propagator->SyncPopulation();

		// Check for Pre-specified Events
		definedList = environmentPopulation.GenerateDefinedEventList();

		// Validate Modules

		ProgressBar progress(simulationDays, '=');

		// --------------------------
		// --- Evolve Environment ---
		// --------------------------

		elapsedDays = 0;
		// While timeSimulation < timeEnd
		while (elapsedDays < simulationDays)
		{
			// Propagate Object Orbits
			//timeStep = min(min(stepDays, environmentPopulation.GetTimeToNextInitEpoch()), simulationDays - elapsedDays);
			timeStep = min(stepDays, simulationDays - elapsedDays);
			(*propagator).PropagatePopulation(timeStep);
			elapsedDays += timeStep;

			while (!definedList.empty()){
				if (definedList[0].GetEventEpoch() < environmentPopulation.GetEpoch()) {
					breakUp->mainBreakup(environmentPopulation, definedList[0]);
					definedList.erase(definedList.begin());
				}
				else
					break;
			}

			// Determine Events
				// Collision Detection
			if (collisionModel->UseGPU())
				collisionModel->MainCollision_GPU(environmentPopulation, timeStep * secondsDay);
			if (collisionModel->UseParallel())
				collisionModel->MainCollision_P(environmentPopulation, timeStep * secondsDay);
			else
				collisionModel->MainCollision(environmentPopulation, timeStep * secondsDay);
			collisionList = collisionModel->GetNewCollisionList();

			n = collisionList.size();
			// if extra output requested
			if ((collisionConfig["Verbose"].asBool()) && (n > 0)) {

					// Retrieve collision output
					collisionOutput = collisionModel->GetNewCollisionVerbose();

					// Log data
					for (int k = 0; k < n; k++) {

						collisionLog.push_back(make_tuple(j, elapsedDays, make_pair(to_string(environmentPopulation.GetObject(collisionList[k].primaryID).GetNoradID()),
							to_string(environmentPopulation.GetObject(collisionList[k].secondaryID).GetNoradID())),
							collisionOutput[k], collisionList[k].altitude));
					}
					collisionOutput.clear();
				
			}

			if (breakUp) {
				// For each pair in collision list
				for (Event collision : collisionList) {
					target = environmentPopulation.GetObject(collision.primaryID);
					projectile = environmentPopulation.GetObject(collision.secondaryID);

					// determine if collision avoidance occurs
					avoidanceProbability = 1 - (1 - target.GetAvoidanceSuccess()) * (1 - projectile.GetAvoidanceSuccess());
					if (collisionModel->DetermineCollisionAvoidance(avoidanceProbability)) {
						// Update and Log
						collision.CollisionAvoidance();
						environmentPopulation.AddDebrisEvent(collision);
					}
					else {
						// Simulate Fragmentations
						breakUp->mainBreakup(environmentPopulation, collision);
					}
				}

				// Generate Explosions
				explosionList = environmentPopulation.GenerateExplosionList();
				if (!explosionList.empty()) {
					for (Event explosion : explosionList) {
						breakUp->mainBreakup(environmentPopulation, explosion);
					}
				}

				simulationLog.push_back(tuple_cat(make_tuple(j), environmentPopulation.GetPopulationState()));
			}
			collisionList.clear();

			progress.DisplayProgress(elapsedDays);
		}

		cout << "\n";
		// ----------------------------
		// ------ End Simulation ------
		// ----------------------------
		// Save final population
		ouputName = populationFilename + "_#" + to_string(j + 1);
		// Write Logs to output files
		if (collisionConfig["Verbose"].asBool()) {

			WriteCollisionData(ouputName, config, collisionType, collisionConfig, collisionLog);
			collisionLog.clear();
		}
		if (logging) {
			WriteEventData(ouputName, config, collisionType, collisionConfig, propagatorType, propagatorConfig, breakUpType, fragmentationConfig, environmentPopulation.GetEventLog());
			WriteSimulationData(ouputName, config, collisionType, collisionConfig, propagatorType, propagatorConfig, breakUpType, fragmentationConfig, simulationLog);
		}
		simulationLog.clear();

		//collisionModel->SetMOID(++moid);
		collisionModel->SwitchParallelGPU();

	}

    return 0;
}


