#include "stdafx.h"

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
		debris = DebrisObject(parsedObject["TLELine1"].asString(),
							  parsedObject["TLELine2"].asString(),
							  parsedObject["TLELine3"].asString());
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

	for (Json::Value objectParameters : scenario["objects"])
	{
		DebrisObject tempObject(GenerateDebrisObject(objectParameters));
		averageSemiMajorAxis += tempObject.GetElements().semiMajorAxis;
		population.AddDebrisObject(tempObject);
	}

	nObjects = scenario["objects"].size();
	population.SetAverageSMA(averageSemiMajorAxis / nObjects);


	date = scenario["EpochDate"].asString();
	epoch = DateToEpoch(date); //TODO read epoch from file
	population.InitialiseEpoch(epoch);

	// Close File
	cout << " Closing Scenario File..." << endl;
	scenarioFile.close();
}

