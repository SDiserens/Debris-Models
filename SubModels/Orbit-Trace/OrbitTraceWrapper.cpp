// OrbitTrace.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Modules\Collision_Algorithms\OrbitTrace.h"

//#define VLD_FORCE_ENABLE
#include <vld.h>

void RandomiseOrbitOrientations(DebrisPopulation& population);
void WriteCollisionData(ofstream & dataFile, string metaData, DebrisPopulation & objectPopulation, map<pair<long, long>, double>& totalCollisionRates,
	vector<map<pair<long, long>, double>>& collisionRates, vector<map<pair<long, long>, int>>& collisionCount, int scalingPower);
void WriteSystemCollisionData(ofstream & dataFile, string metaData, DebrisPopulation & objectPopulation, map<pair<long, long>, double>& totalCollisionRates,
	vector<map<pair<long, long>, double>>& collisionRates, vector<map<pair<long, long>, int>>& collisionCount, int scalingPower);


int main(int argc, char** argv)
{
	string arg, scenarioFilename, outputFilename, eventType, metaData;
	unsigned long evaluationBlocks, evaluationSteps;
	uint64_t seed, argseed = -1;
	int scalingPower, MOID=0;
	bool probabilityOutput, relativeGravity, printing, individualOutput, randomiseOrbits, OTfilters, GPU, CPU;
	double timeStepDays, timeStep, scaling, threshold;
	char date[100];
	int ID = 1;
	Json::Value config, scenario, parsedObject;

	LoadConfigFile(config);

	// Identify config variables
	scenarioFilename = config["scenarioFilename"].asString();
	probabilityOutput = config["probabilityOutput"].asBool();
	relativeGravity = config["relativeGravity"].asBool();
	randomiseOrbits = config["randomiseOrbits"].asBool();
	evaluationBlocks = config["numberEvaluations"].asUInt();
	evaluationSteps = config["stepsPerEvaluation"].asUInt();
	timeStepDays = config["stepSize"].asDouble();
	printing = config["outputPrinting"].asBool();
	individualOutput = config["individualOutput"].asBool();
	OTfilters = config["filters"].asBool();
	threshold = config["ConjunctionThreshold"].asDouble();
	timeStep = timeStepDays * secondsDay;
	GPU = config["ParallelGPU"].asBool();
	CPU = config["ParallelCPU"].asBool();
	MOID = config["MOID"].asInt();

	// Parse command line arguments
	for (int i = 1; i < argc; ++i) {
		arg = argv[i];
		if ((arg == "-f") || (arg == "--filename"))
		{
			scenarioFilename = argv[++i];
		}
		else if ((arg == "-b") || (arg == "--blocks"))
		{
			evaluationBlocks = stoi(argv[++i]);
		}
		else if ((arg == "-s") || (arg == "--steps"))
		{
			evaluationSteps = stoi(argv[++i]);
		}
		else if ((arg == "-t") || (arg == "--threshold"))
		{
			threshold = stod(argv[++i]);
		}
		else if ((arg == "-v") || (arg == "--verbose"))
		{
			printing = true;
		}
		else if (arg == "--seed")
		{
			argseed = stoi(argv[++i]);
		}
		else if (arg == "--CPU")
		{
			CPU = true;
		}
		else if (arg == "--GPU")
		{
			GPU = true;
		}
	}


	// Create population of objects & Identify average SMA
	DebrisPopulation objectPopulation;
	LoadScenario(objectPopulation, scenarioFilename);

	scalingPower = objectPopulation.GetScalingPower();
	scaling = pow(10, scalingPower);

	// Run simulation
	if (config["randomSeed"].isUInt64() || (argseed != -1))
	{
		seed = (argseed != -1) ? argseed : config["randomSeed"].asUInt64();
		cout << "Using a random seed of : " << seed << endl;
		SeedRNG(seed);
	}

	// Create OT object
	OrbitTrace collisionModel(probabilityOutput, threshold, MOID);
	if (GPU)
		collisionModel.SwitchParallelGPU();
	if (CPU)
		collisionModel.SwitchParallelCPU();
	if (relativeGravity)
		collisionModel.SwitchGravityComponent();
	if (!OTfilters)
		collisionModel.SwitchFilters();

	unsigned int step, eval;
	double tempCollisionRate, blockRatio;
	vector<Event> collisionList;
	pair<long, long> pairID;
	map<pair<long, long>, double> totalCollisionRates;
	map<pair<long, long>, int> totalCollisionCount;
	vector<map<pair<long, long>, double>> collisionRates;
	vector<map<pair<long, long>, int>> collisionCount;
	collisionCount.resize(evaluationBlocks);
	collisionRates.resize(evaluationBlocks);

	blockRatio = secondsYear / (evaluationSteps * timeStep);
	// Call CUBE approach
	ProgressBar progress(evaluationBlocks * evaluationSteps, '=');
	cout << "Using " + to_string(evaluationBlocks) + " blocks of " + to_string(evaluationSteps) + " steps." << endl;


	// Call cube algorithm
	auto start = std::chrono::system_clock::now(); // starttime
	auto end = std::chrono::system_clock::now(); //endtime
	std::chrono::duration<double> timeDiff = end - end;

	for (eval = 0; eval < evaluationBlocks; eval++)
	{
		for (step = 0; step < evaluationSteps; step++)
		{
			start = std::chrono::system_clock::now();
			//Randomise variables
			if (randomiseOrbits)
				RandomiseOrbitOrientations(objectPopulation);
			//Call Collision check
			if (collisionModel.UseGPU())
				collisionModel.MainCollision_GPU(objectPopulation, timeStep);
			else if (collisionModel.UseParallel())
				collisionModel.MainCollision_P(objectPopulation, timeStep);
			else
				collisionModel.MainCollision(objectPopulation, timeStep);
			end = std::chrono::system_clock::now();
			timeDiff += end - start;
			progress.DisplayProgress(eval * evaluationSteps + step);
		}
		// Store collision data
		collisionList = collisionModel.GetNewCollisionList();

		for (Event collision : collisionList)
		{
			tempCollisionRate = scaling * collision.collisionProbability * blockRatio;
			pairID = collision.GetCollisionPair();
			totalCollisionRates[pairID] = totalCollisionRates[pairID] + tempCollisionRate;
			totalCollisionCount[pairID] = totalCollisionCount[pairID] + 1;
			collisionRates[eval][pairID] = collisionRates[eval][pairID] + tempCollisionRate;
			collisionCount[eval][pairID] = collisionCount[eval][pairID] + 1;
		}
		collisionList.clear();
		collisionList.shrink_to_fit();
	}

	progress.DisplayProgress(evaluationBlocks * evaluationSteps); cout << "\n" << flush;

	string collisionName;

	cout << "Calculated in runtime of " << timeDiff.count() << "s\n" << endl;

	for (auto & collisionPair : totalCollisionRates)
	{
		pairID = collisionPair.first;
		tempCollisionRate = round(1e4 * (collisionPair.second / evaluationBlocks)) / 1e4;
		collisionPair.second = tempCollisionRate;
		collisionName = objectPopulation.GetObject(pairID.first).GetName() + "-" +
			objectPopulation.GetObject(pairID.second).GetName();
		if (printing)
		{
			cout << "For collision pair: " + collisionName + ":\n" << flush;
			cout << "-- Collision rate = " + to_string(collisionPair.second) + " * 10^-" + to_string(scalingPower) + " per year.\n" +
				" Based on " + to_string(totalCollisionCount[pairID]) + " conjunctions.\n" << flush;
		}
	}

	// Store data
	time_t dateTime = time(NULL);
	struct tm currtime;
	localtime_s(&currtime, &dateTime);
	strftime(date, sizeof(date), "%F", &currtime);

	eventType = scenarioFilename.substr(0, scenarioFilename.find("."));

	outputFilename = "Output\\" + string(date) + "_" + eventType + ".csv";
	while (fileExists(outputFilename))
	{
		ID++;
		outputFilename = "Output\\" + string(date) + "_" + eventType + '_' + to_string(ID) + ".csv";
	}

	cout << "Creating Data File : " + outputFilename + "...";
	// Create Output file
	ofstream outputFile;
	outputFile.open(outputFilename, ofstream::out);

	// Write data into file
	cout << "  Writing to Data File...";

	metaData = "Scenario : ," + eventType +
		"\nNumber of evaluations : ," + to_string(evaluationBlocks) + "\nEvaluation Steps : ," + to_string(evaluationSteps) + "\nStep Length : ," + to_string(timeStep) + ",seconds\n" +
		"Using a scaling of : ," + to_string(scaling) + "\nCalculated in runtime of : ," + to_string(timeDiff.count()) + ",s";

	if (individualOutput)
		WriteCollisionData(outputFile, metaData, objectPopulation, totalCollisionRates, collisionRates, collisionCount, scalingPower);
	else
		WriteSystemCollisionData(outputFile, metaData, objectPopulation, totalCollisionRates, collisionRates, collisionCount, scalingPower);

	cout << "Finished\n";
	// Close file
	outputFile.close();

	// END
	return 0;
}

void RandomiseOrbitOrientations(DebrisPopulation& population)
{
	double rAAN, argP;
	for (auto& debris : population.population)
	{
		//	-- Generate random orientation (randomTau)
		rAAN = randomNumberTau();
		argP = randomNumberTau();
		//Update object
		debris.second.UpdateRAAN(rAAN);
		debris.second.UpdateArgP(argP);
	}
}

void WriteSystemCollisionData(ofstream & dataFile, string metaData, DebrisPopulation & objectPopulation, map<pair<long, long>, double>& totalCollisionRates,
	vector<map<pair<long, long>, double>>& collisionRates, vector<map<pair<long, long>, int>>& collisionCount, int scalingPower)
{
	/*
	MetaData = [['Simulation:', 'Jovian_Moons'], ['Cube Dimension', Dim, 'km'], ['Number of sections, N:', N], ['Section length, n:', n, 'days']]
	*/

	dataFile << metaData + "\n";
	dataFile << "\n";

	unsigned int i, tempCount = 0;
	double tempRate = 0;
	pair<long, long> pairID;

	dataFile << "Total Collision Rate";
	for (auto& collisionPair : totalCollisionRates)
	{
		tempRate += collisionPair.second;
	}
	dataFile << ',' + to_string(tempRate) + ",* 10 ^ -" + to_string(scalingPower) + " per year.\n";


	dataFile << "\n";
	dataFile << "Block, Count, Rate\n";
	map<pair<long, long>, double> blockRates;
	map<pair<long, long>, int> blockCounts;

	for (i = 0; i < collisionRates.size(); i++)
	{
		blockRates = collisionRates[i];
		blockCounts = collisionCount[i];
		tempCount = 0;
		tempRate = 0;
		dataFile << "Collision Block " + to_string(i + 1) + ",";
		for (auto& collisionPair : totalCollisionRates)
		{
			pairID = collisionPair.first;
			tempRate += blockRates[pairID];
			tempCount += blockCounts[pairID];
		}

		dataFile << to_string(tempCount) + ',' + to_string(tempRate) + ',';
		dataFile << " * 10 ^ -" + to_string(scalingPower) + " per year.\n";
	}
}

void WriteCollisionData(ofstream & dataFile, string metaData, DebrisPopulation & objectPopulation, map<pair<long, long>, double>& totalCollisionRates,
	vector<map<pair<long, long>, double>>& collisionRates, vector<map<pair<long, long>, int>>& collisionCount, int scalingPower)
{
	// Determine output format
	/*
	MetaData = [['Simulation:', 'Jovian_Moons'], ['Cube Dimension', Dim, 'km'], ['Number of sections, N:', N], ['Section length, n:', n, 'days']]
	*/

	dataFile << metaData + "\n";
	dataFile << "\n";
	/*
	Collision Pairs :-
	Total Collision Rate per pair :-
	Total Conjunction Rate per pair :-
	----
	Rate per evaluation step for each pair
	----
	*/

	// Write data to file
	unsigned int i;
	double tempRate;
	string collisionName, collisionID;
	pair<long, long> pairID;

	dataFile << "Collision IDs";
	for (auto const& collisionPair : totalCollisionRates)
	{
		pairID = collisionPair.first;
		collisionID = "(" + to_string(pairID.first) + "-" + to_string(pairID.second) + ")";
		dataFile << ',' + collisionID;
	}

	dataFile << "\nCollision Pair";
	for (auto& collisionPair : totalCollisionRates)
	{
		pairID = collisionPair.first;
		collisionName = objectPopulation.GetObject(pairID.first).GetName() + "-" +
			objectPopulation.GetObject(pairID.second).GetName();
		dataFile << ',' + collisionName;
	}

	dataFile << "\nTotal Collision Rate";
	for (auto& collisionPair : totalCollisionRates)
	{

		dataFile << ',' + to_string(collisionPair.second);
	}
	dataFile << ",* 10 ^ -" + to_string(scalingPower) + " per year.\n";

	// Break data with line
	dataFile << "\n";

	map<pair<long, long>, double> blockRates;
	map<pair<long, long>, int> blockCounts;

	for (i = 0; i < collisionRates.size(); i++)
	{

		blockRates = collisionRates[i];

		dataFile << "Collision Rate - Block " + to_string(i + 1) + ",";
		for (auto& collisionPair : totalCollisionRates)
		{
			pairID = collisionPair.first;
			tempRate = blockRates[pairID];
			dataFile << to_string(tempRate) + ',';
		}

		dataFile << " * 10 ^ -" + to_string(scalingPower) + " per year.\n";
	}

	// Break data with line
	dataFile << "\n";

	for (i = 0; i < collisionCount.size(); i++)
	{

		blockCounts = collisionCount[i];
		dataFile << "Collision Count - Block " + to_string(i + 1) + ",";
		for (auto& collisionPair : totalCollisionRates)
		{
			pairID = collisionPair.first;
			tempRate = blockCounts[pairID];
			dataFile << to_string(tempRate) + ',';
		}

		dataFile << "\n";
	}
}


