#include "stdafx.h"

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


DebrisObject GenerateDebrisObject(Json::Value & parsedObject, double epoch)
{
	Json::Value elements;
	double radius, mass, length, semiMajorAxis, eccentricity, inclination, rightAscension, argPerigee, meanAnomaly;
	int type, dataType;
	string name;
	DebrisObject debris;

	if (parsedObject.isMember("dataType"))
		dataType = parsedObject["dataType"].asInt();
	else
		dataType = 0;

	switch(dataType)
	{
	case 0:
		elements = parsedObject["orbitalElements"];
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
		debris = DebrisObject(radius, mass, length, semiMajorAxis, eccentricity, inclination, rightAscension, argPerigee, meanAnomaly, type);
		debris.SetName(name);
		debris.SetInitEpoch(epoch);
		break;

	case 1:
		throw std::runtime_error("STate VEctor initialisation not implemented");
		break;

	case 2:

		radius = parsedObject["radius"].asDouble();
		debris = DebrisObject(parsedObject["TLELine1"].asString(),
							  parsedObject["TLELine2"].asString(),
							  parsedObject["TLELine3"].asString());
		debris.SetRadius(radius);
		break;
	}

	return debris;
}

void LoadConfigFile(Json::Value & config)
{
	Json::Reader reader;

	cout << "Reading Config File...\n";
	// Read config file
	ifstream configFile("config.json");

	// Parse config file to identify scenario file and settings
	cout << " Parsing Config...\n";
	reader.parse(configFile, config);

	// Close File
	cout << " Closing Config File...\n";
	configFile.close();
}

void LoadScenario(DebrisPopulation & population, string scenarioFilename)
{
	population.Clear();
	Json::Value config, scenario, parsedObject;
	Json::Reader reader;
	int nObjects;
	double averageSemiMajorAxis = 0;
	double epoch;
	string date;

	// Read scenario file

	ifstream scenarioFile("Scenarios\\" + scenarioFilename);
	if (!scenarioFile.good())
	{
		cout << "Scenario file failed to load";
		throw std::runtime_error("Scenario file failed to load");
	}

	// Parse scenario file to identify object characteristics
	reader.parse(scenarioFile, scenario);

	cout << " Parsing Scenario...\n";
	SetCentralBody(scenario["centralBody"].asInt());
	population.SetScalingPower(scenario["outputScaling"].asInt());

	date = scenario["EpochDate"].asString();
	epoch = DateToEpoch(date);
	population.InitialiseEpoch(epoch);
	population.SetDuration(scenario["Duration"].asDouble());

	for (Json::Value objectParameters : scenario["objects"])
	{
		DebrisObject tempObject(GenerateDebrisObject(objectParameters, epoch));
		averageSemiMajorAxis += tempObject.GetElements().semiMajorAxis;
		population.AddDebrisObject(tempObject);
	}

	nObjects = scenario["objects"].size();
	population.SetAverageSMA(averageSemiMajorAxis / nObjects);
			
	// Close File
	cout << " Closing Scenario File...\n" << endl;
	scenarioFile.close();
}

void WriteCollisionData(string scenario, Json::Value & config, string collisionModel, Json::Value & collisionConfig, vector<tuple<int, double, pair<string, string>, double, double>> collisionLog)
{
	char date[100];
	int ID = 1;
	string outputFilename, pairID, mcRun;
	double scaling;
	// Store data
	time_t dateTime = time(NULL);
	struct tm currtime;
	localtime_s(&currtime, &dateTime);
	strftime(date, sizeof(date), "%F", &currtime);


	mcRun = scenario.substr(scenario.find("#") - 1, scenario.find("."));
	scenario = scenario.substr(0, scenario.find("#") - 1);

	outputFilename = "Output\\" + string(date) + scenario + "_CollisionData_" + mcRun + ".csv";
	while (fileExists(outputFilename))
	{
		ID++;
		outputFilename = "Output\\" + string(date) + scenario + "_" + to_string(ID) + "_CollisionData_" + mcRun + ".csv";
	}

	cout << "Creating Data File : " + outputFilename + "...";
	// Create Output file
	ofstream outputFile;
	outputFile.open(outputFilename, ofstream::out);

	// Write data into file
	cout << "  Writing to Data File...";

	outputFile << "Scenario File:," + scenario;
	outputFile << "\nDuration:," + config["Duration"].asString() + ",Days"; // Length of simulation (days)
	outputFile << "\nStep Length:," + config["StepSize"].asString() + ",Days";
	outputFile << "\nCollision Model:," + collisionModel;

	if (collisionConfig.isMember("outputScaling")) {
		int scalingPower = collisionConfig["outputScaling"].asInt();
		scaling = pow(10, scalingPower);
		outputFile << "\nScaling Power:," + to_string(scalingPower);
	}
	else
		scaling = 1;

	if (collisionModel == "Cube")
		outputFile << "\nCube Dimension (km):," + to_string(collisionConfig["CubeDimension"].asDouble());
	if (collisionModel == "OrbitTrace")
		outputFile << "\nThreshold (km):," + to_string(collisionConfig["ConjunctionThreshold"].asDouble());
	if (collisionModel == "Hoots")
	{
		outputFile << "\nConjunction Threshold (km):," + to_string(collisionConfig["ConjunctionThreshold"].asDouble());
		outputFile << "\nCollision Threshold (km):," + to_string(collisionConfig["CollisionThreshold"].asDouble());
	}

	// Break data with line
	outputFile << "\n";

	outputFile << "\nSimulation Run, Simulation Elapsed Time (days), Object Pair, Collision Probability, Altitude (km)";
	for (auto logEntry : collisionLog) 
	{
		pairID = "'" + get<2>(logEntry).first + " - " + get<2>(logEntry).second;
		outputFile << "\n" + to_string(get<0>(logEntry)) + "," + to_string(get<1>(logEntry)) + "," + pairID + "," + to_string(scaling * get<3>(logEntry)) + "," + to_string(get<4>(logEntry));
	}

}

void WriteSimulationData(string scenario, Json::Value & config, string collisionModel, Json::Value & collisionConfig, string propagatorType, Json::Value & propagatorConfig, string breakUpType,
						Json::Value & fragmentationConfig, vector<tuple<int, double, int, tuple<int, int, int>, int, tuple<int, int, int>>> simulationLog)
{
	char date[100];
	int ID = 1;
	string outputFilename, pairID, mcRun;
	tuple<int, int, int> eventSplit, objectSplit;
	// Store data
	time_t dateTime = time(NULL);
	struct tm currtime;
	localtime_s(&currtime, &dateTime);
	strftime(date, sizeof(date), "%F", &currtime);


	mcRun = scenario.substr(scenario.find("#") - 1, scenario.find("."));
	scenario = scenario.substr(0, scenario.find("#") - 1);

	outputFilename = "Output\\" + string(date) + scenario + "_SimulationData_" + mcRun + ".csv";
	while (fileExists(outputFilename))
	{
		ID++;
		outputFilename = "Output\\" + string(date) + scenario + "_" + to_string(ID) + "_SimulationData_" + mcRun + ".csv";
	}


	cout << "Creating Data File : " + outputFilename + "...";
	// Create Output file
	ofstream outputFile;
	outputFile.open(outputFilename, ofstream::out);

	// Write data into file
	cout << "  Writing to Data File...";

	outputFile << "Scenario File:," + scenario + ", ,";
	outputFile << "Duration:," + config["Duration"].asString() + ",Days" + ", ,"; // Length of simulation (days)
	outputFile << ",Step Length:," + config["StepSize"].asString() + ",Days" + "\n";
	outputFile << "Collision Model:," + collisionModel + ", ,";
	outputFile << "Fragmentation Model:," + breakUpType + ", ,";
	outputFile << "Propagator:," + propagatorType;

	if (collisionModel == "Cube")
		outputFile << "\nCube Dimension (km):," + to_string(collisionConfig["CubeDimension"].asDouble());
	if (collisionModel == "OrbitTrace")
		outputFile << "\nThreshold (km):," + to_string(collisionConfig["ConjunctionThreshold"].asDouble());
	if (collisionModel == "Hoots")
	{
		outputFile << "\nConjunction Threshold (km):," + to_string(collisionConfig["ConjunctionThreshold"].asDouble());
		outputFile << "\nCollision Threshold (km):," + to_string(collisionConfig["CollisionThreshold"].asDouble());
	}
	outputFile << "\nMinimum Fragment Size (m):," + to_string(fragmentationConfig["minLength"].asDouble());

	// Break data with line
	outputFile << "\n";

	outputFile << "\nSimulation Run, Simulation Elapsed Time (days), Object Count, -UpperStage Count, -Spacecraft Count, -Debris Count, Event Count, -Explosion Count, -Collision Count, -Collision Avoidance Count"; // (MC, #days, #objects, (), #events, (Explosion, Collision, Collision Avoidance))
	for (auto logEntry : simulationLog)
	{
		eventSplit = get<5>(logEntry);
		objectSplit = get<3>(logEntry);
		outputFile << "\n" + to_string(get<0>(logEntry)) + "," + to_string(get<1>(logEntry)) + "," + to_string(get<2>(logEntry)) + to_string(get<0>(objectSplit)) + "," + to_string(get<1>(objectSplit)) + "," + to_string(get<2>(objectSplit));
		outputFile << to_string(get<4>(logEntry)) + "," + to_string(get<0>(eventSplit)) + "," + to_string(get<1>(eventSplit)) + "," + to_string(get<2>(eventSplit));
	}
}

void WriteEventData(string scenario, Json::Value & config, string collisionModel, Json::Value & collisionConfig, string propagatorType, Json::Value & propagatorConfig, string breakUpType, Json::Value & fragmentationConfig, vector<Event> eventLog)
{
	char date[100];
	int ID = 1;
	string outputFilename, pairID, mcRun;
	tuple<int, int, int> eventSplit;
	// Store data
	time_t dateTime = time(NULL);
	struct tm currtime;
	localtime_s(&currtime, &dateTime);
	strftime(date, sizeof(date), "%F", &currtime);

	mcRun = scenario.substr(scenario.find("#")-1, scenario.find("."));
	scenario = scenario.substr(0, scenario.find("#")-1);

	outputFilename = "Output\\" + string(date) + scenario + "_EventData_" + mcRun + ".csv";
	while (fileExists(outputFilename))
	{
		ID++;
		outputFilename = "Output\\" + string(date) + scenario + "_" + to_string(ID) + "_EventData_"  + mcRun + ".csv";
	}

	cout << "Creating Data File : " + outputFilename + "...";
	// Create Output file
	ofstream outputFile;
	outputFile.open(outputFilename, ofstream::out);

	// Write data into file
	cout << "  Writing to Data File...";

	outputFile << "Scenario File:," + scenario + ", ,";
	outputFile << "Duration:," + config["Duration"].asString() + ",Days" + ", ,"; // Length of simulation (days)
	outputFile << ",Step Length:," + config["StepSize"].asString() + ",Days" + "\n";
	outputFile << "Collision Model:," + collisionModel + ", ,";
	outputFile << "Fragmentation Model:," + breakUpType + ", ,";
	outputFile << "Propagator:," + propagatorType;

	if (collisionModel == "Cube")
		outputFile << "\nCube Dimension (km):," + to_string(collisionConfig["CubeDimension"].asDouble());
	if (collisionModel == "OrbitTrace")
		outputFile << "\nThreshold (km):," + to_string(collisionConfig["ConjunctionThreshold"].asDouble());
	if (collisionModel == "Hoots")
	{
		outputFile << "\nConjunction Threshold (km):," + to_string(collisionConfig["ConjunctionThreshold"].asDouble());
		outputFile << "\nCollision Threshold (km):," + to_string(collisionConfig["CollisionThreshold"].asDouble());
	}
	outputFile << "\nMinimum Fragment Size (m):," + to_string(fragmentationConfig["minLength"].asDouble());

	// Break data with line
	outputFile << "\n";

	outputFile << "\nEvent ID, Simulation Elapsed Time (days), Event Type ID, Event Type, Primary Object, Secondary Object, Debris Count, Altitude, Involved Mass, Relative Velocity, Catastrophic, Momentum Conserved"; // (MC, #days, #objects, (), #events, (Explosion, Collision, Collision Avoidance))
	for (auto logEntry : eventLog)
	{
		outputFile << "\n" + to_string(logEntry.eventID) + "," + to_string(logEntry.GetEventEpoch()) + "," + to_string(logEntry.GetEventType()) + "," + logEntry.GetEventTypeString() + ",";
		outputFile << to_string(logEntry.GetPrimary()) + "," + to_string(logEntry.GetSecondary()) + "," + to_string(logEntry.debrisGenerated) + "," + to_string(logEntry.altitude) + ",";
		outputFile << to_string(logEntry.involvedMass) + "," +  to_string(logEntry.relativeVelocity) + "," + to_string(logEntry.catastrophic) + "," + to_string(logEntry.momentumConserved);
	}
}

