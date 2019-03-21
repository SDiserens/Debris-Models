// CUBE.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Modules/Collision_Algorithms/CUBE.h"
#include <json\json.h>




void RandomiseOrbitOrientations(DebrisPopulation& population);
void WriteCollisionData(ofstream & dataFile, string metaData, DebrisPopulation & objectPopulation, map<pair<long, long>, double>& totalCollisionRates,
						vector<map<pair<long, long>, double>>& collisionRates, vector<map<pair<long, long>, int>>& collisionCount, int scalingPower);
void WriteSystemCollisionData(ofstream & dataFile, string metaData, DebrisPopulation & objectPopulation, map<pair<long, long>, double>& totalCollisionRates,
	vector<map<pair<long, long>, double>>& collisionRates, vector<map<pair<long, long>, int>>& collisionCount, int scalingPower);
DebrisObject GenerateDebrisObject(Json::Value & parsedObject);
bool fileExists(const string& name);


int main(int argc, char** argv)
{

	string arg, scenarioFilename, outputFilename, eventType, metaData;
	uint64_t evaluationBlocks, evaluationSteps, seed, argseed = -1;
	int runMode, scalingPower, nObjects;
	bool probabilityOutput, relativeGravity, printing, individualOutput;
	double timeStepDays, timeStep, dimension, cubeDimension, scaling;
	double averageSemiMajorAxis = 0;

	char date[100];
	int ID = 1;
	Json::Value config, scenario, parsedObject;
	Json::Reader reader;

	cout << "Reading Config File...";
	// Read config file
	ifstream configFile("config.json");
	// Parse config file to identify scenario file and settings
	reader.parse(configFile, config);
	cout << " Parsing Config...";

	// Identify config variables
	scenarioFilename =  config["scenarioFilename"].asString();
	probabilityOutput = config["probabilityOutput"].asBool();
	relativeGravity = config["relativeGravity"].asBool();
	runMode = config["runType"].asInt();
	dimension = config["cubeDimension"].asDouble();
	evaluationBlocks = config["numberEvaluations"].asUInt64();
	evaluationSteps = config["stepsPerEvaluation"].asUInt64();
	timeStepDays = config["stepSize"].asDouble();
	printing = config["outputPrinting"].asBool();
	individualOutput = config["individualOutput"].asBool();
	timeStep = timeStepDays * secondsDay;

	// Parse command line arguments
	for (int i = 1; i < argc; ++i) {
		arg = argv[i];
		if ((arg == "-f") || (arg == "--filename"))
		{
			scenarioFilename = argv[++i];
		} 
		else if ((arg == "-d") || (arg == "--dimension"))
		{
			dimension = atof(argv[++i]);
		}
		else if ((arg == "-b") || (arg == "--blocks"))
		{
			evaluationBlocks = stoi(argv[++i]);
		}
		else if ((arg == "-s") || (arg == "--steps"))
		{
			evaluationSteps = stoi(argv[++i]);
		}
		else if ((arg == "-t") || (arg == "--time"))
		{
			timeStepDays = atof(argv[++i]);
		}
		else if ((arg == "-v") || (arg == "--verbose"))
		{
			printing = true;
		}
		else if (arg == "--seed")
		{
			argseed = stoi(argv[++i]);
		}
	}

	// Close File
	cout << " Closing Config File...\n";
	configFile.close();

	// Read scenario file
	cout << "Reading Scenario File : " + scenarioFilename + "...";

	ifstream scenarioFile("Scenarios\\" + scenarioFilename);
	if (!scenarioFile.good())
	{
		throw std::runtime_error("Scenario file failed to load");
	}

	// Parse scenario file to identify object characteristics
	reader.parse(scenarioFile, scenario);

	cout << " Parsing Scenario...";
	SetCentralBody(scenario["centralBody"].asInt());
	scalingPower = scenario["outputScaling"].asInt();
	scaling = pow(10, scalingPower);

	// Create population of objects & Identify average SMA
	DebrisPopulation objectPopulation;
	for (Json::Value objectParameters : scenario["objects"])
	{
		DebrisObject tempObject(GenerateDebrisObject(objectParameters));
		averageSemiMajorAxis += tempObject.GetElements().semiMajorAxis;
		objectPopulation.AddDebrisObject(tempObject);
	}
	nObjects = scenario["objects"].size();
	averageSemiMajorAxis /= nObjects;
	cubeDimension = averageSemiMajorAxis * dimension;

	for (int i = 1; i < argc; ++i) {
		arg = argv[i];
		if ((arg == "-c") || (arg == "--cubesize"))
		{
			cubeDimension = atof(argv[++i]);
		}
	}
	// Close File
	cout << " Closing Scenario File..." << endl;
	scenarioFile.close();

	// Run simulation
	if (config["randomSeed"].isUInt64() || (argseed != -1) )
	{
		seed = (argseed != -1) ? argseed : config["randomSeed"].asUInt64();
		cout << "Using a random seed of : " << seed << endl;
		SeedRNG(seed);
	}

	// Create Cube object
	CUBEApproach collisionCube(cubeDimension, probabilityOutput);
	if (relativeGravity)
		collisionCube.SwitchGravityComponent();

	int step, eval, k;
	double tempCollisionRate, blockRatio;
	vector<double> collisionProbabilities;
	vector<pair<long, long>> collisionList;
	map<pair<long, long>, double> totalCollisionRates;
	map<pair<long, long>, int> totalCollisionCount;
	vector<map<pair<long, long>, double>> collisionRates;
	vector<map<pair<long, long>, int>> collisionCount;
	collisionCount.resize(evaluationBlocks);
	collisionRates.resize(evaluationBlocks);

	blockRatio = secondsYear / (evaluationSteps * timeStep);
	// Call CUBE approach
	ProgressBar progress(evaluationBlocks * evaluationSteps, '=');
	cout << "Using a Cube Length of " + to_string(cubeDimension) + "km and " + to_string(evaluationBlocks) + " blocks of " + to_string(evaluationSteps) + " steps." << endl;


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
			RandomiseOrbitOrientations(objectPopulation);
			//Call Collision check
			collisionCube.MainCollision(objectPopulation, timeStep);
			end = std::chrono::system_clock::now();
			timeDiff += end - start;
			progress.DisplayProgress(eval * evaluationSteps + step);
		}
		// Store collision data
		collisionProbabilities = collisionCube.GetNewCollisionProbabilities();
		collisionList = collisionCube.GetNewCollisionList();

		for (k = 0; k < collisionProbabilities.size(); k++)
		{
			tempCollisionRate = scaling * collisionProbabilities[k] * blockRatio;
			totalCollisionRates[collisionList[k]] = totalCollisionRates[collisionList[k]] + tempCollisionRate;
			totalCollisionCount[collisionList[k]] = totalCollisionCount[collisionList[k]] + 1;
			collisionRates[eval][collisionList[k]] = collisionRates[eval][collisionList[k]] + tempCollisionRate;
			collisionCount[eval][collisionList[k]] = collisionCount[eval][collisionList[k]] + 1;
		}
	}

	progress.DisplayProgress(evaluationBlocks * evaluationSteps); cout << "\n" << flush;
	
	pair<long, long> pairID;
	string collisionName;


	cout << "Calculated in runtime of " << timeDiff.count() << "s\n" << endl;

	for (auto & collisionPair : totalCollisionRates)
	{
		pairID = collisionPair.first;
		tempCollisionRate = round(1e4 * (collisionPair.second/ evaluationBlocks)) / 1e4;
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

	metaData = "Scenario : ," + eventType + "\nDimension : ," + to_string(100 * dimension) + ",% of average semiMajorAxis\n Cube Dimension : ," + to_string(cubeDimension) + ",km\n" + 
				"Number of evaluations : ," + to_string(evaluationBlocks) + "\nEvaluation Steps : ," + to_string(evaluationSteps) + "\nStep Length : ," + to_string(timeStep) + ",seconds\n" +
				"Using a scaling of : ," + to_string(scaling) + "\nCalculated in runtime of : ," + to_string(timeDiff.count()) + ",s";

	if (individualOutput)
		WriteCollisionData(outputFile, metaData, objectPopulation, totalCollisionRates, collisionRates, collisionCount, scalingPower);
	else
		WriteSystemCollisionData(outputFile, metaData, objectPopulation, totalCollisionRates, collisionRates, collisionCount, scalingPower);

	cout << "Finished\n";
	// Close file
	outputFile.close();

	// END
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

DebrisObject GenerateDebrisObject(Json::Value & parsedObject)
{
	double radius, mass, length, semiMajorAxis, eccentricity, inclination, rightAscension, argPerigee, meanAnomaly;
	int type;
	string name;

	Json::Value elements = parsedObject["orbitalElements"];
	// Parse Json 
	radius = parsedObject["radius"].asDouble();
	mass = parsedObject["mass"].asDouble();
	length = parsedObject["length"].asDouble();
	meanAnomaly = parsedObject["meanAnomaly"].asDouble();
	type = parsedObject["type"].asInt();
	semiMajorAxis = elements["a"].asDouble();
	eccentricity = elements["e"].asDouble();
	inclination = elements["i"].asDouble();
	rightAscension = elements["Om"].asDouble();
	argPerigee = elements["om"].asDouble();
	name = parsedObject["name"].asString();
	// Generate Object - Possible issue with reconstruction
	DebrisObject debris(radius, mass, length, semiMajorAxis, eccentricity, inclination, rightAscension, argPerigee, meanAnomaly, type);
	debris.SetName(name);
	return debris;
}

void WriteSystemCollisionData(ofstream & dataFile, string metaData, DebrisPopulation & objectPopulation, map<pair<long, long>, double>& totalCollisionRates,
	vector<map<pair<long, long>, double>>& collisionRates, vector<map<pair<long, long>, int>>& collisionCount, int scalingPower)
{
	/* 
	MetaData = [['Simulation:', 'Jovian_Moons'], ['Cube Dimension', Dim, 'km'], ['Number of sections, N:', N], ['Section length, n:', n, 'days']]
	*/
	
	dataFile << metaData + "\n";
	dataFile << "\n";
	
	int i, tempCount=0;
	double tempRate=0;
	pair<long, long> pairID;

	dataFile << "Total Collision Rate";
	for (auto& collisionPair : totalCollisionRates)
	{
		tempRate += collisionPair.second;
	}
	dataFile << ',' + to_string(tempRate)+ ",* 10 ^ -" + to_string(scalingPower) + " per year.\n";
	

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
	int i;
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

		dataFile << "Collision Rate - Block " + to_string(i+1) + ",";
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

bool fileExists(const string& name)
{
	FILE *file;
	fopen_s(&file, name.c_str(), "r");
	if (file)
	{
		fclose(file);
		return true;
	}
	else
		return false;
}


