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


DebrisObject GenerateDebrisObject(Json::Value & parsedObject)
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

void LoadScenario(DebrisPopulation & population, string scenarioFilename)
{
	Json::Value config, scenario, parsedObject;
	Json::Reader reader;
	int nObjects;
	double averageSemiMajorAxis = 0;
	double epoch;
	string date;

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
	population.SetScalingPower(scenario["outputScaling"].asInt());

	date = scenario["EpochDate"].asString();
	epoch = DateToEpoch(date);
	population.InitialiseEpoch(epoch);

	for (Json::Value objectParameters : scenario["objects"])
	{
		DebrisObject tempObject(GenerateDebrisObject(objectParameters));
		averageSemiMajorAxis += tempObject.GetElements().semiMajorAxis;
		population.AddDebrisObject(tempObject);
	}

	nObjects = scenario["objects"].size();
	population.SetAverageSMA(averageSemiMajorAxis / nObjects);
			
	// Close File
	cout << " Closing Scenario File..." << endl;
	scenarioFile.close();
}

void WriteCollisionData(string scenario, Json::Value & config, string collisionModel, Json::Value & collisionConfig, vector<tuple<double, pair<long, long>, double>> collisionLog)
{
	char date[100];
	int ID = 1;
	string outputFilename, eventType, pairID;

	// Store data
	time_t dateTime = time(NULL);
	struct tm currtime;
	localtime_s(&currtime, &dateTime);
	strftime(date, sizeof(date), "%F", &currtime);

	eventType = scenario.substr(0, scenario.find("."));

	outputFilename = "Output\\" + string(date) + "_CollisionData_" + eventType + ".csv";
	while (fileExists(outputFilename))
	{
		ID++;
		outputFilename = "Output\\" + string(date) + "_CollisionData_" + eventType + "_" + to_string(ID) + ".csv";
	}

	cout << "Creating Data File : " + outputFilename + "...";
	// Create Output file
	ofstream outputFile;
	outputFile.open(outputFilename, ofstream::out);

	// Write data into file
	cout << "  Writing to Data File...";

	outputFile << "Scenario File:," + scenario;
	outputFile << "\nDuration:" + config["Duration"].asString(); // Length of simulation (days)
	outputFile << "\nStep Length:" + config["StepSize"].asString();
	outputFile << "\nCollision Model:," + collisionModel;
	if (collisionModel == "Cube")
		outputFile << "\nCube Dimension (km):," + config["CubeDimension"].asString();
	if (collisionModel == "OrbitTrace")
		outputFile << "\nThreshold (km):," + config["conjunctionThreshold"].asString();
	if (collisionModel == "Hoots")
	{
		outputFile << "\nConjunction Threshold (km):," + config["conjunctionThreshold"].asString();
		outputFile << "\nCollision Threshold (km):," + config["collisionThreshold"].asString();
	}

	// Break data with line
	outputFile << "\n";

	outputFile << "\nSimulation Elapsed Time (days), Object Pair, Collision Probability";
	for (auto logEntry : collisionLog) 
	{
		//TODO - Add scaling to collision so not stored as 0
		pairID = to_string(get<1>(logEntry).first) + "-" + to_string(get<1>(logEntry).second);
		outputFile << "\n" + to_string(get<0>(logEntry)) + "," + pairID + "," + to_string(get<2>(logEntry));
	}

}

