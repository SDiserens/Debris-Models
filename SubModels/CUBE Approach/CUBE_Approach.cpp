// CUBE.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Modules/Collision_Algorithms/CUBE.h"
#include <json\json.h>




void RandomiseOrbitOrientations(DebrisPopulation& population);
void WriteCollisionData(ofstream & dataFile, string metaData, map<pair<long, long>, double> totalCollisionRates, vector<map<pair<long, long>, double>> collisionRates,
						vector<map<pair<long, long>, int>> collisionCount, int scalingPower);
DebrisObject GenerateDebrisObject(Json::Value & parsedObject);
bool fileExists(const string& name);


int main()
{
	string scenarioFilename, outputFilename, eventType, metaData;
	int evaluationBlocks, evaluationSteps, runMode, scalingPower, nObjects;
	bool probabilityOutput, relativeGravity;
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
	dimension = config["cubeDimension"].asDouble();

	runMode = config["runType"].asInt();
	evaluationBlocks = config["numberEvaluations"].asInt();
	evaluationSteps = config["stepsPerEvaluation"].asInt();
	timeStepDays = config["stepSize"].asDouble();
	timeStep = timeStepDays * secondsDay;

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

	// Close File
	cout << " Closing Scenario File...\n";
	scenarioFile.close();

	// Run simulation
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

	blockRatio = 1 / (evaluationSteps * timeStep / secondsYear);
	// Call CUBE approach
	ProgressBar progress(evaluationBlocks * evaluationSteps, '=');
	cout << "Using a Cube Length of " + to_string(cubeDimension) + "km and " + to_string(evaluationBlocks) + " blocks of " + to_string(evaluationSteps) + " steps.\n" << flush;

	for (eval = 0; eval < evaluationBlocks; eval++)
	{
		for (step = 0; step < evaluationSteps; step++)
		{
			//Randomise variables
			RandomiseOrbitOrientations(objectPopulation);
			//Call Collision check
			collisionCube.MainCollision(objectPopulation, timeStep);
			progress.DisplayProgress(eval * evaluationSteps + step);
		}
		// TODO - Store collision data
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

	for (auto & collisionPair : totalCollisionRates)
	{
		pairID = collisionPair.first;
		tempCollisionRate = round(1e4 * (collisionPair.second/ evaluationBlocks)) / 1e4;
		collisionPair.second = tempCollisionRate;
		collisionName = objectPopulation.GetObject(pairID.first).GetName() + "-" +
						objectPopulation.GetObject(pairID.second).GetName();
		cout << "For collision pair: " + collisionName + ":\n" << flush;
		cout << "-- Collision rate = " + to_string(collisionPair.second) + " * 10^-" + to_string(scalingPower) + " per year.\n" + 
				" Based on " + to_string(totalCollisionCount[pairID]) + " conjunctions.\n" << flush;
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

	metaData = "Scenario : ," + eventType + "\nDimension : ," + to_string(100 * dimension) + ",% of average semiMajorAxis\n Cube Dimension : ," + to_string(cubeDimension) + "km \n" + 
				"Number of evaluations - N: ," + to_string(evaluationBlocks) + ",\nEvaluation Steps : ," + to_string(evaluationSteps) + ",\nStep Length : ," + to_string(timeStep) + ",days\n" +
				"Using a scaling of," + to_string(scaling);
	WriteCollisionData(outputFile, metaData, totalCollisionRates, collisionRates, collisionCount, scalingPower);

	cout << "Finished\n";
	// Close file
	outputFile.close();

	// END
}

void RandomiseOrbitOrientations(DebrisPopulation& population)
{
	double rAAN, argP;
	for (pair<long, DebrisObject> debris : population.population)
	{
		//	-- Generate random orientation (randomTau)
		rAAN = randomNumberTau();
		argP = randomNumberTau();
		//Update object
		debris.second.UpdateRAAN(rAAN);
		debris.second.UpdateRAAN(argP);
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
	name = parsedObject["name"].asString();;
	// Generate Object - Possible issue with reconstruction
	DebrisObject debris(radius, mass, length, semiMajorAxis, eccentricity, inclination, rightAscension, argPerigee, meanAnomaly, type);
	debris.SetName(name);
	return debris;
}

void WriteCollisionData(ofstream & dataFile, string metaData, map<pair<long, long>, double> totalCollisionRates, vector<map<pair<long, long>, double>> collisionRates,
						vector<map<pair<long, long>, int>> collisionCount, int scalingPower)
{
	// Determine output format
	/* 
	MetaData = [['Simulation:', 'Jovian_Moons'], ['Cube Dimension', Dim, 'km'], ['Number of sections, N:', N], ['Section length, n:', n, 'days']]
	*/
	
	dataFile << metaData + "\n";

	/*
	Collision Pairs :-
	Total Collision Rate per pair :-
	Total Conjunction Rate per pair :-
	----
	Rate per evaluation step for each pair
	----
	*/

	// TODO - Write data to file
	int i;
	double tempRate;
	string collisionName;
	pair<long, long> pairID;

	dataFile << "Collision Pair,";
	for (auto const& collisionPair : totalCollisionRates)
	{
		pairID = collisionPair.first;
		collisionName = to_string(pairID.first) + "-" + to_string(pairID.second);
		dataFile << collisionName + ',';
	}

	dataFile << "\nTotal Collision Rate,";
	for (auto const& collisionPair : totalCollisionRates)
	{

		dataFile << to_string(collisionPair.second) + ',';
	}
	dataFile << " * 10 ^ -" + to_string(scalingPower) + " per year.\n";

	dataFile << "\n";
	for (i = 0; i < collisionRates.size(); i++)
	{

		dataFile << "Collision Block " + to_string(i+1) + ",";
		for (auto const& collisionPair : totalCollisionRates)
		{
			pairID = collisionPair.first;
			tempRate = collisionRates[i][pairID];
			dataFile << to_string(tempRate) + ',';
		}

		dataFile << " * 10 ^ -" + to_string(scalingPower) + " per year.\n";
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


