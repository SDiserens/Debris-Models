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

DebrisObject GenerateDebrisObject(string line)
{
	Json::Value elements;
	double diameter, area, radius, mass, semiMajorAxis, eccentricity, inclination, rightAscension, argPerigee, meanAnomaly, launchDate;
	int type, type2;

	istringstream iss(line);
	//#  Type Mass Diameter Area a e i RAAN AoP M LA-DATE    
	iss >> type >> mass >> diameter >> area >> semiMajorAxis >> eccentricity >> inclination >> rightAscension >> argPerigee >> meanAnomaly >> launchDate;

	switch (type) {
	case 1: type2 = 0;
	case 2: type2 = 1;
	case 3: type2 = 2;
	case 4: type2 = 2;
	}

	radius = sqrt(area / Pi);
	DebrisObject tempObject(radius, mass, diameter, semiMajorAxis, eccentricity, inclination, rightAscension, argPerigee, meanAnomaly, type2);
	tempObject.SetInitEpoch(launchDate);
	return tempObject;
}


vector<DebrisObject> GenerateLaunchTraffic(Json::Value & launches)
{
	vector<DebrisObject> launchTraffic;
	double launchEpoch;
	DebrisObject tempObject;
	// ToDo - Update to read from launch .pop file
	if (launches.isArray()) {
		for (Json::Value objectParameters : launches)
		{
			launchEpoch = objectParameters["LaunchDate"].asDouble();
			tempObject = GenerateDebrisObject(objectParameters, launchEpoch);
			launchTraffic.push_back(tempObject);
		}
	}
	else if (launches.isString())
	{
		string line;
		string scenarioFilename = launches.asString();
		ifstream launchFile("Environment\\Launch_Traffic\\" + scenarioFilename);
		if (!launchFile.good())
		{
			cout << "Launch file failed to load";
			throw std::runtime_error("Launch file failed to load");
		}
		while (getline(launchFile, line)) {
			if (line.at(0) != '#') {
				tempObject = GenerateDebrisObject(line);
				launchTraffic.push_back(tempObject);
			}
		}
	}
	sort(launchTraffic.begin(), launchTraffic.end(), CompareInitEpochs);
	return launchTraffic;
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
	Json::Value config, scenario, parsedObject, definedEvent;
	Json::Reader reader;
	int nObjects, collisionID;
	double averageSemiMajorAxis = 0;
	double epoch, mass;
	string date;
	DebrisObject tempObject;
	map<int, long> definedCollisions;

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
	if (scenario["objects"].isArray()) {
		for (Json::Value objectParameters : scenario["objects"])
		{
			tempObject = GenerateDebrisObject(objectParameters, epoch);
			averageSemiMajorAxis += tempObject.GetElements().semiMajorAxis;
			population.AddDebrisObject(tempObject);

			// Load any pre-defined events
			if (objectParameters.isMember("definedEvent")) {
				definedEvent = objectParameters["definedEvent"];
				if (!definedEvent.isMember("CollisionID")) {
					// Load explosion
					Event tempEvent(definedEvent["epoch"].asDouble(), tempObject.GetID(), tempObject.GetMass());
					population.AddDefinedEvent(tempEvent);
				}
				else {
					// Store collision data
					collisionID = definedEvent["CollisionID"].asInt();
					if (definedCollisions.count(collisionID) > 0) {
						// Check if other collision object exists yet
						mass = tempObject.GetMass() + population.GetObject(definedCollisions[collisionID]).GetMass();
						Event tempEvent(definedEvent["epoch"].asDouble(), definedCollisions[collisionID], tempObject.GetID(), definedEvent["RelativeVelocity"].asDouble(), mass, definedEvent["Altitude"].asDouble());
						population.AddDefinedEvent(tempEvent);
					}
					else
						// Store object ID for later
						definedCollisions[collisionID] = tempObject.GetID();
				}


			}
		}
	}
	else if (scenario["objects"].isString())
	{
		string line;
		string scenarioFilename = scenario["objects"].asString();
		ifstream populationFile("Environment\\Background_Population\\" + scenarioFilename);
		if (!populationFile.good())
		{
			cout << "Population file failed to load";
			throw std::runtime_error("Population file failed to load");
		}
		while (getline(populationFile, line)) {
			if (line.at(0) != '#') {
				tempObject = GenerateDebrisObject(line);
				population.AddDebrisObject(tempObject);
			}
		}
	}


	double launchCyle = scenario["LaunchCycle"].asDouble();	
	population.AddLaunchTraffic(GenerateLaunchTraffic(scenario["launches"]), launchCyle);

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
