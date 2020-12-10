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
	double radius, area, mass, length, semiMajorAxis, eccentricity, inclination, rightAscension, argPerigee, meanAnomaly;
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
		if (parsedObject.isMember("area"))
			debris.SetArea(parsedObject["area"].asDouble());
		debris.SetName(name);
		debris.SetInitEpoch(epoch);
		debris.SetEpoch(epoch);
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
		if (parsedObject.isMember("mass"))
			debris.SetMass(parsedObject["mass"].asDouble());
		break;
	}

	return debris;
}

DebrisObject GenerateDebrisObject(string line)
{
	Json::Value elements;
	double diameter, area, radius, mass, semiMajorAxis, eccentricity, inclination, rightAscension, argPerigee, meanAnomaly, launchDate;
	int type, type2=2;

	istringstream iss(line);
	//#  Type Mass Diameter Area a e i RAAN AoP M LA-DATE    
	iss >> type >> mass >> diameter >> area >> semiMajorAxis >> eccentricity >> inclination >> rightAscension >> argPerigee >> meanAnomaly >> launchDate;

	switch (type) {
	case 1: 
		type2 = 0;
		break;
	case 2: 
		type2 = 1;
		break;
	case 3: 
		type2 = 2;
		break;
	case 4: 
		type2 = 2;
		break;
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
	if (launches.isArray()) {
		for (Json::Value objectParameters : launches)
		{
			launchEpoch = objectParameters["LaunchDate"].asDouble();
			tempObject = GenerateDebrisObject(objectParameters, launchEpoch);
			launchTraffic.push_back(tempObject);
		}
	}
	//read from launch .pop file
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
	Json::Value scenario;
	Json::Reader reader;
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
	//SetCentralBody(scenario["centralBody"].asInt());
	population.SetScalingPower(scenario["outputScaling"].asInt());

	date = scenario["EpochDate"].asString();
	epoch = DateToEpoch(date);
	population.InitialiseEpoch(epoch);
	population.SetDuration(scenario["Duration"].asDouble());

	LoadObjects(population, scenario);

	double launchCyle = scenario["LaunchCycle"].asDouble();	
	//population.AddLaunchTraffic(GenerateLaunchTraffic(scenario["launches"]), launchCyle);

			
	// Close File
	cout << " Closing Scenario File...\n" << endl;
	scenarioFile.close();
}

void LoadBackground(DebrisPopulation & population, string backgroundFilename) {
	Json::Value scenario;
	Json::Reader reader;

	ifstream backgroundFile("Scenarios\\" + backgroundFilename);
	if (!backgroundFile.good())
	{
		cout << "Background file failed to load";
		throw std::runtime_error("Background file failed to load");
	}

	// Parse scenario file to identify object characteristics
	reader.parse(backgroundFile, scenario);


	LoadObjects(population, scenario);
	cout << " Closing Background File...\n" << endl;
	backgroundFile.close();
}

void LoadObjects(DebrisPopulation & population, Json::Value scenario) {
	DebrisObject tempObject, newObject;
	Json::Value parsedObject, definedEvent;
	double cycle, averageSemiMajorAxis = 0;
	int nObjects, collisionID, body, i;
	map<int, long> definedCollisions;
	double epoch, mass, start;
	vector<DebrisObject> launches;

	if (scenario.isMember("launchCycle")) {
		cycle = scenario["launchCycle"].asDouble() * 365.25;
	}
	else {
		cycle = 8 * 365.25;
	}

	epoch = population.GetEpoch();
	if (scenario.isMember("centralBody"))
		body = scenario["centralBody"].asInt();
	else
		body = 3;
	if (scenario["objects"].isArray()) {

		for (Json::Value objectParameters : scenario["objects"])
		{
			tempObject = GenerateDebrisObject(objectParameters, epoch);
			tempObject.SetCentralBody(body);
			if (objectParameters.isMember("constellationID"))
				tempObject.SetConstellationID(objectParameters["constellationID"].asInt());
			averageSemiMajorAxis += tempObject.GetElements().semiMajorAxis;

			// Load laucnh objects
			if (objectParameters.isMember("LaunchDate")) {
				tempObject.SetLaunchCycle(cycle);
				newObject = CopyDebrisObject(tempObject);
				start = DateToEpoch(objectParameters["LaunchDate"].asDouble());
				i = 0;
				while (start < population.GetStartEpoch()) {
					start += cycle;
					i = 1;
				}
				start = start - i*cycle;
				tempObject.SetInitEpoch(start);
				newObject.SetInitEpoch(start + cycle);
				launches.push_back(newObject);
			}

			long ID = population.AddDebrisObject(tempObject);

			// Load any pre-defined events
			if (objectParameters.isMember("definedEvent")) {
				definedEvent = objectParameters["definedEvent"];
				if (definedEvent.isMember("CollisionID"))
				{
					// Store collision data
					collisionID = definedEvent["CollisionID"].asInt();
					if (definedCollisions.count(collisionID) > 0) {
						// Check if other collision object exists yet
						mass = tempObject.GetMass() + population.GetObject(definedCollisions[collisionID]).GetMass();
						Event tempEvent(epoch + definedEvent["fromEpoch"].asDouble(), definedCollisions[collisionID], ID, definedEvent["RelativeVelocity"].asDouble(), mass, 0.0, 0.0, 1.0);
						population.AddDefinedEvent(tempEvent);
					}
					else
						// Store object ID for later
						definedCollisions[collisionID] = ID;
				}
				else {
					Event tempEvent(epoch + definedEvent["fromEpoch"].asDouble(), ID, tempObject.GetMass());
					population.AddDefinedEvent(tempEvent);

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
	nObjects = scenario["objects"].size();
	population.SetAverageSMA(averageSemiMajorAxis / nObjects);
	population.AddLaunchTraffic(launches);
}

void WriteCollisionData(string scenario, Json::Value & config, string collisionModel, Json::Value & collisionConfig, vector<tuple<int, double, pair<string, string>, double, double, double>> collisionLog)
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

	outputFilename = "Output\\" + string(date) + "_" + scenario + "_CollisionData" + mcRun + ".csv";
	while (fileExists(outputFilename))
	{
		++ID;
		outputFilename = "Output\\" + string(date) + "_" + scenario + "_" + to_string(ID) + "_CollisionData" + mcRun + ".csv";
	}

	cout << "- Creating Data File : " + outputFilename + "...";
	// Create Output file
	ofstream outputFile;
	outputFile.open(outputFilename, ofstream::out);

	// Write data into file
	cout << "  Writing to Data File...";

	outputFile << "Scenario File:," + scenario;
	//outputFile << "\nDuration:," + config["Duration"].asString() + ",Days"; // Length of simulation (days)
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

	outputFile << "\nSimulation Run, Simulation Elapsed Time (days), Object Pair, Collision Probability, Altitude (km), Minimum Distance (km)";
	for (auto logEntry : collisionLog) 
	{
		pairID = "'" + get<2>(logEntry).first + " - " + get<2>(logEntry).second;
		outputFile << "\n" + to_string(get<0>(logEntry)) + "," + to_string(get<1>(logEntry)) + "," + pairID + "," + to_string(scaling * get<3>(logEntry)) + "," + to_string(get<4>(logEntry)) + "," + to_string(get<5>(logEntry));
	}
	outputFile.close();
	cout << "\n";
}

void WriteSimulationData(string scenario, Json::Value & config, double epoch, string collisionModel, Json::Value & collisionConfig, string propagatorType, Json::Value & propagatorConfig, string breakUpType,
						Json::Value & fragmentationConfig, vector<tuple<int, double, int, tuple<int, int, int>, int, tuple<int, int, int, int>>> simulationLog)
{
	char date[100];
	int ID = 1;
	string outputFilename, pairID, mcRun;
	tuple<int, int, int, int> eventSplit;
	tuple<int, int, int> objectSplit;
	// Store data
	time_t dateTime = time(NULL);
	struct tm currtime;
	localtime_s(&currtime, &dateTime);
	strftime(date, sizeof(date), "%F", &currtime);


	mcRun = scenario.substr(scenario.find("#") - 1, scenario.find("."));
	scenario = scenario.substr(0, scenario.find("#") - 1);

	outputFilename = "Output\\" + string(date) + "_" + scenario + "_SimulationData" + mcRun + ".csv";
	while (fileExists(outputFilename))
	{
		++ID;
		outputFilename = "Output\\" + string(date) + "_" + scenario + "_" + to_string(ID) + "_SimulationData" + mcRun + ".csv";
	}


	cout << "- Creating Data File : " + outputFilename + "...";
	// Create Output file
	ofstream outputFile;
	outputFile.open(outputFilename, ofstream::out);

	// Write data into file
	cout << "  Writing to Data File...";

	outputFile << "Scenario File:," + scenario + ", ," + EpochToDate(epoch) + "\n";
	//outputFile << "Duration:," + config["Duration"].asString() + ",Days" + ", ,"; // Length of simulation (days)

	outputFile << "Propagator:," + propagatorType + ", ,";
	outputFile << "Step Length:," + config["StepSize"].asString() + ",Days" + "\n";

	outputFile << "Collision Model:," + collisionModel + ", ,";

	if (collisionModel == "Cube" || collisionModel == "Cube - new"	)
		outputFile << "Cube Dimension (km):," + to_string(collisionConfig["CubeDimension"].asDouble()) + "\n";
	if (collisionModel == "OrbitTrace" || collisionModel == "OrbitTrace - new")
		outputFile << "Threshold (km):," + to_string(collisionConfig["ConjunctionThreshold"].asDouble()) + "\n";
	if (collisionModel == "Hoots")
	{
		outputFile << "Conjunction Threshold (km):," + to_string(collisionConfig["ConjunctionThreshold"].asDouble()) + ", ,";
		outputFile << "Collision Threshold (km):," + to_string(collisionConfig["CollisionThreshold"].asDouble()) + "\n";
	}

	outputFile << "Fragmentation Model:," + breakUpType + ", ,";
	outputFile << "Minimum Fragment Size (m):," + to_string(fragmentationConfig["minLength"].asDouble());

	// Break data with line
	outputFile << "\n";

	outputFile << "\nSimulation Run, Simulation Elapsed Time (days), Object Count, -UpperStage Count, -Spacecraft Count, -Debris Count, Event Count, -Explosion Count, -Collision Count, -Collision Avoidance Count, (-Catastrophic Collision Count)"; // (MC, #days, #objects, (), #events, (Explosion, Collision, Collision Avoidance, Catastrophic collision))
	for (auto logEntry : simulationLog)
	{
		eventSplit = get<5>(logEntry);
		objectSplit = get<3>(logEntry);
		outputFile << "\n" + to_string(get<0>(logEntry)) + "," + to_string(get<1>(logEntry) - epoch) + ",";
		outputFile << to_string(get<2>(logEntry)) + "," + to_string(get<0>(objectSplit)) + "," + to_string(get<1>(objectSplit)) + "," + to_string(get<2>(objectSplit)) + ",";
		outputFile << to_string(get<4>(logEntry)) + "," + to_string(get<0>(eventSplit)) + "," + to_string(get<1>(eventSplit)) + "," + to_string(get<2>(eventSplit)) + "," + to_string(get<3>(eventSplit));
	}
	outputFile.close();
	cout << "\n";
}

void WriteEventData(string scenario, Json::Value & config, double epoch, string collisionModel, Json::Value & collisionConfig, string propagatorType, Json::Value & propagatorConfig, string breakUpType, Json::Value & fragmentationConfig, vector<Event> eventLog)
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

	outputFilename = "Output\\" + string(date) + "_" + scenario + "_EventData" + mcRun + ".csv";
	while (fileExists(outputFilename))
	{
		++ID;
		outputFilename = "Output\\" + string(date) + "_" + scenario + "_" + to_string(ID) + "_EventData"  + mcRun + ".csv";
	}

	cout << "- Creating Data File : " + outputFilename + "...";
	// Create Output file
	ofstream outputFile;
	outputFile.open(outputFilename, ofstream::out);

	// Write data into file
	cout << "  Writing to Data File...";

	outputFile << "Scenario File:," + scenario + ", ," + EpochToDate(epoch) + "\n";
	//outputFile << "Duration:," + config["Duration"].asString() + ",Days" + ", ,"; // Length of simulation (days)

	outputFile << "Propagator:," + propagatorType + ", ,";
	outputFile << "Step Length:," + config["StepSize"].asString() + ",Days" + "\n";

	outputFile << "Collision Model:," + collisionModel + ", ,";

	if (collisionModel == "Cube")
		outputFile << "Cube Dimension (km):," + to_string(collisionConfig["CubeDimension"].asDouble()) + "\n";
	if (collisionModel == "OrbitTrace")
		outputFile << "Threshold (km):," + to_string(collisionConfig["ConjunctionThreshold"].asDouble()) + "\n";
	if (collisionModel == "Hoots")
	{
		outputFile << "Conjunction Threshold (km):," + to_string(collisionConfig["ConjunctionThreshold"].asDouble()) + ", ,";
		outputFile << "Collision Threshold (km):," + to_string(collisionConfig["CollisionThreshold"].asDouble()) + "\n";
	}

	outputFile << "Fragmentation Model:," + breakUpType + ", ,";
	outputFile << "Minimum Fragment Size (m):," + to_string(fragmentationConfig["minLength"].asDouble());

	// Break data with line
	outputFile << "\n";

	outputFile << "\nEvent ID, Simulation Elapsed Time (days), Event Type ID, Event Type, Primary Object, Secondary Object, Debris Count, Altitude, Involved Mass, Relative Velocity, Catastrophic, Momentum Conserved"; // (MC, #days, #objects, (), #events, (Explosion, Collision, Collision Avoidance))
	for (auto logEntry : eventLog)
	{
		outputFile << "\n" + to_string(logEntry.eventID) + "," + to_string(logEntry.GetEventEpoch() - epoch) + "," + to_string(logEntry.GetEventType()) + "," + logEntry.GetEventTypeString() + ",";
		outputFile << to_string(logEntry.GetPrimary()) + "," + to_string(logEntry.GetSecondary()) + "," + to_string(logEntry.debrisGenerated) + "," + to_string(logEntry.altitude) + ",";
		outputFile << to_string(logEntry.involvedMass) + "," +  to_string(logEntry.relativeVelocity) + "," + to_string(logEntry.catastrophic) + "," + to_string(logEntry.momentumConserved);
	}

	outputFile.close();
	cout << "\n";
}

void WritePopulationData(string scenario, Json::Value & config, DebrisPopulation & population, string collisionModel, Json::Value & collisionConfig, string propagatorType, Json::Value & propagatorConfig, string breakUpType,
	Json::Value & fragmentationConfig)
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
	double epoch = population.GetStartEpoch();
	DebrisObject tempDebris;
	RemovedObject removeDebris;

	mcRun = scenario.substr(scenario.find("#") - 1, scenario.find("."));
	scenario = scenario.substr(0, scenario.find("#") - 1);

	outputFilename = "Output\\" + string(date) + "_" + scenario + "_PopulationData" + mcRun + ".csv";
	while (fileExists(outputFilename))
	{
		++ID;
		outputFilename = "Output\\" + string(date) + "_" + scenario + "_" + to_string(ID) + "_PopulationData" + mcRun + ".csv";
	}

	cout << "- Creating Data File : " + outputFilename + "...";
	// Create Output file
	ofstream outputFile;
	outputFile.open(outputFilename, ofstream::out);

	// Write data into file
	cout << "  Writing to Data File...";

	outputFile << "Scenario File:," + scenario + ", ," + EpochToDate(epoch) + "\n";
	//outputFile << "Duration:," + config["Duration"].asString() + ",Days" + ", ,"; // Length of simulation (days)

	outputFile << "Propagator:," + propagatorType + ", ,";
	outputFile << "Step Length:," + config["StepSize"].asString() + ",Days" + "\n";

	outputFile << "Collision Model:," + collisionModel + ", ,";

	if (collisionModel == "Cube")
		outputFile << "Cube Dimension (km):," + to_string(collisionConfig["CubeDimension"].asDouble()) + "\n";
	if (collisionModel == "OrbitTrace")
		outputFile << "Threshold (km):," + to_string(collisionConfig["ConjunctionThreshold"].asDouble()) + "\n";
	if (collisionModel == "Hoots")
	{
		outputFile << "Conjunction Threshold (km):," + to_string(collisionConfig["ConjunctionThreshold"].asDouble()) + ", ,";
		outputFile << "Collision Threshold (km):," + to_string(collisionConfig["CollisionThreshold"].asDouble()) + "\n";
	}	

	outputFile << "Fragmentation Model:," + breakUpType + ", ,";
	outputFile << "Minimum Fragment Size (m):," + to_string(fragmentationConfig["minLength"].asDouble());

	// Break data with line
	outputFile << "\n";

	outputFile << "\nObject ID, Name, Object Type, Source Event, Mass, Remove Event, Remove Epoch, ParentID, SourceID,  Source Type, Fragments Represented, State, Lifetime Collision Probability";
		for (auto object : population.removedPopulation) {
			removeDebris = object.second;
			outputFile << "\n" + to_string(object.first) + "," + removeDebris.GetName() + "," + to_string(removeDebris.GetType()) + "," + to_string(removeDebris.GetSourceEvent()) +","+to_string(removeDebris.GetMass());
			outputFile << "," + to_string(removeDebris.GetRemoveEvent()) + "," + to_string(removeDebris.GetRemoveEpoch() - epoch);
			outputFile << "," + to_string(removeDebris.GetParentID()) + "," + to_string(removeDebris.GetSourceID()) + "," + to_string(removeDebris.GetSourceType()) + "," + to_string(removeDebris.GetNFrag());
			if (removeDebris.GetType() == 2)
				outputFile << ",Debris";
			else if (removeDebris.IsActive())
				outputFile << ",Active";
			else if (removeDebris.IsPassive())
				outputFile << ",Inactive - Passivated";
			else if (removeDebris.IsIntact())
				outputFile << ",Inactive - Not Passivated";
			else
				outputFile << ",Inactive-Damaged";
			outputFile << "," + to_string(removeDebris.GetCollisionProbability());
		}
		for (auto object : population.population) {
			tempDebris = object.second;
			outputFile << "\n" + to_string(object.first) + "," + tempDebris.GetName() + "," + to_string(tempDebris.GetType()) + "," + to_string(tempDebris.GetSourceEvent()) + "," + to_string(tempDebris.GetMass());
			outputFile << ", , ";
			outputFile << "," + to_string(tempDebris.GetParentID()) + "," + to_string(tempDebris.GetSourceID()) + "," + to_string(tempDebris.GetSourceType()) + "," + to_string(tempDebris.GetNFrag());
			if (tempDebris.GetType() == 2)
				outputFile << ",Debris";
			else if(tempDebris.IsActive())
				outputFile << ",Active";
			else if (tempDebris.IsPassive())
				outputFile << ",Inactive - Passivated";
			else if (tempDebris.IsIntact())
				outputFile << ",Inactive - Not Passivated";
			else
				outputFile << ",Inactive-Damaged";
			outputFile << "," + to_string(tempDebris.GetCollisionProbability());
		}

		outputFile.close();
		cout << "\n";
}
