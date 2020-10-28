// Framework.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Modules\Modules.h"
#include "Utilities\ModuleFactory.h"
//#define VLD_FORCE_ENABLE
#include <vld.h>
using namespace std;

int main(int argc, char** argv)
{
	// ----------------------------
	// ------ Initialisation ------
	// ----------------------------

	// Config objects
	Json::Value config, propagatorConfig, fragmentationConfig, collisionConfig;

	// Variable
	string arg, populationFilename, propagatorType, breakUpType, collisionType, ouputName, backgroundFilename;
	double timeStep, stepDays, elapsedDays, simulationDays, threshold, tempProbability, avoidanceProbability = 0;
	bool  launches = false, logging = true, setThreshold = false;
	int mcRuns, i;
	volatile int n;

	// Data logs
	vector<tuple<int, double, int, tuple<int, int, int>, int, tuple<int, int, int>>> simulationLog;
	// (MC, #days, #objects, (upperstage, spacecraft, debris), #events, (Explosion, Collision, Collision Avoidance)) 
	vector<tuple<int, double, pair<string, string>, double, double, double>> collisionLog;
	// (MC, #days, objectIDs, probability, altitude)
	vector<Event> collisionList;
	vector<Event> explosionList;
	vector<Event> definedList;

	// ----------------------------
	// - Parsing config variables -
	// ----------------------------
	LoadConfigFile(config);

	populationFilename = config["scenarioFilename"].asString();
	if (config.isMember("BackgroundPop")) 
		backgroundFilename = config["BackgroundPop"].asString();
	mcRuns = config["MonteCarlo"].asInt();
	logging = config["logging"].asBool();
	stepDays = config["StepSize"].asDouble();
	launches = config["launches"].asBool();

	propagatorType = config["Propagator"].asString();
	propagatorConfig = config["PropagatorConfig"];

	breakUpType = config["Fragmentation"].asString();
	fragmentationConfig = config["FragmentationConfig"];

	collisionType = config["CollsionAlgorithm"].asString();
	collisionConfig = config["CollisionConfig"];
	int moid = collisionConfig["MOIDtype"].asInt();

	//Set explosion probabilities
	rocketBodyExplosionProbability = config["RB_Explosion"].asDouble();
	satelliteExplosionProbability = config["Satllite_Explosion"].asDouble();
	manoeuvreThreshold = config["CA_Manoeuvre_Threshold"].asDouble();

	//Set pmd success
	pmdSuccess = config["PMD_SuccessRate"].asDouble();
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
		if ((arg == "-s") || (arg == "--stepsize"))
		{
			stepDays = stod(argv[++i]);
		}
		if ((arg == "-l") || (arg == "--logging"))
		{
			logging = stod(argv[++i]);
		}
	}

	// ----------------------------
	// --- Initialise population --
	// ----------------------------
	DebrisPopulation environmentPopulation, initPopulation;

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

	cout << "Reading Population File : " + populationFilename + "...\n";
	LoadScenario(initPopulation, populationFilename);
	if (config.isMember("BackgroundPop")) {
		cout << "Reading Background File : " + backgroundFilename + "...\n";
		LoadBackground(initPopulation, backgroundFilename);
	}
	populationFilename = populationFilename.substr(0, populationFilename.find("."));
	simulationDays = initPopulation.GetDuration();
	initPopulation.SetLaunches(launches);

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
					Event tempEvent = definedList[0];
					if (tempEvent.GetEventType()) {
						CollisionPair objectPair(environmentPopulation.GetObject(tempEvent.primaryID), environmentPopulation.GetObject(tempEvent.secondaryID));
						objectPair.GenerateArgumenstOfIntersection();
						tempEvent.SetCollisionAnomalies(objectPair.approachAnomalyP, objectPair.approachAnomalyS);
						tempEvent.SetAltitude(objectPair.primaryElements.GetRadialPosition());
						objectPair.~CollisionPair();
					}
					breakUp->mainBreakup(environmentPopulation, tempEvent);
					definedList.erase(definedList.begin());
					tempEvent.~Event();
				}
				else
					break;
			}

			// Determine Events
				// Collision Detection
			if (collisionModel->UseGPU())
				collisionModel->MainCollision_GPU(environmentPopulation, timeStep * secondsDay);
			else if (collisionModel->UseParallel())
				collisionModel->MainCollision_P(environmentPopulation, timeStep * secondsDay);
			else
				collisionModel->MainCollision(environmentPopulation, timeStep * secondsDay);
			collisionList = collisionModel->GetNewCollisionList();

			n = collisionList.size();
			// if extra output requested
			if ((collisionConfig["Verbose"].asBool()) && (n > 0)) {
					// Log data
					for (Event collision : collisionList) {
						collisionLog.push_back(make_tuple(j, elapsedDays, make_pair(to_string(environmentPopulation.GetObject(collision.primaryID).GetNoradID()),
							to_string(environmentPopulation.GetObject(collision.secondaryID).GetNoradID())),
							collision.collisionProbability, collision.altitude, collision.minSeparation));
					}
			}

			// Simulate Breakups and collision avoidance
			if (breakUp) {
				shuffle(collisionList.begin(), collisionList.end(), *generator);
				// For each pair in collision list
				for (Event collision : collisionList) {
					if (!environmentPopulation.CheckObject(collision.primaryID) || !environmentPopulation.CheckObject(collision.secondaryID))
						continue;
					DebrisObject& target = environmentPopulation.GetObject(collision.primaryID);
					DebrisObject& projectile = environmentPopulation.GetObject(collision.secondaryID);
					tempProbability = 1.0 - pow((1.0 - collision.collisionProbability), (target.GetNFrag() * projectile.GetNFrag())); // adjust probabiltiy for representative fragments
					//if (isnan(tempProbability))
					//	tempProbability = 0;
					collision.collisionProbability = tempProbability;

					// determine if collision avoidance occurs
					avoidanceProbability = 1 - (1 - target.GetAvoidanceSuccess()) * (1 - projectile.GetAvoidanceSuccess());

					if (!collisionModel->CheckValidCollision(target, projectile)) {
						collision.InvalidCollision();
						environmentPopulation.AddDebrisEvent(collision);
					}
					else {
						target.UpdateCollisionProbability(tempProbability);
						projectile.UpdateCollisionProbability(tempProbability);

						if ((collision.collisionProbability >= manoeuvreThreshold) & (collisionModel->DetermineCollisionAvoidance(avoidanceProbability))
							& (target.GetLength() > 0.1)& (projectile.GetLength() > 0.1)) {
							// Update and Log
							collision.CollisionAvoidance();
							environmentPopulation.AddDebrisEvent(collision);
						}

						else if (collisionModel->DetermineCollision(tempProbability)) {
							// Simulate Fragmentations
							breakUp->mainBreakup(environmentPopulation, collision);
						}
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
			WriteEventData(ouputName, config, environmentPopulation.GetStartEpoch(), collisionType, collisionConfig, propagatorType, propagatorConfig, breakUpType, fragmentationConfig, environmentPopulation.GetEventLog());
			WriteSimulationData(ouputName, config, environmentPopulation.GetStartEpoch(), collisionType, collisionConfig, propagatorType, propagatorConfig, breakUpType, fragmentationConfig, simulationLog);
			WritePopulationData(ouputName, config, environmentPopulation, collisionType, collisionConfig, propagatorType, propagatorConfig, breakUpType, fragmentationConfig);
		}
		simulationLog.clear();

		//collisionModel->SetMOID(++moid);
	}

    return 0;
}


