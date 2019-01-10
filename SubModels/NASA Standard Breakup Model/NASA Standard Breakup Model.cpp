// NASA Standard Breakup Model.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "../../Modules/Fragmentation_Models/NSBM.h"
#include <json\json.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <ctime>

void WritePopulationData(ofstream & dataFile, DebrisPopulation & population, DebrisObject& targetObject, DebrisObject *projectilePointer);
DebrisObject GenerateDebrisObject(Json::Value & parsedObject);
bool fileExists(const string& name);

double minLength;

int main()
{
	string scenarioFilename, outputFilename, eventType;
	char date[100];
	int ID = 1;
	Json::Value config, scenario, parsedObject;
	Json::Reader reader;

	// Read config file
	ifstream configFile("config.json");

	// Parse config file to identify scenario file and settings
	reader.parse(configFile, config);

	double minLength = config["minLength"].asDouble();
	scenarioFilename = "Scenarios\\" + config["scenarioFilename"].asString();
	numFragBuckets = config["numberOfBuckets"].asInt();
	bridgingFunction = config["bridgingFunction"].asString();
	catastrophicThreshold = config["catastrophicThreshold"].asDouble();
	//TODO - debug assignment of global variables!

	// Close File
	configFile.close();

	// Read scenario file
	ifstream scenarioFile(scenarioFilename);
	if (!scenarioFile.good())
	{
		throw std::runtime_error("Scenario file failed to load");
	}

	// Parse scenario file to identify object characteristics
	reader.parse(scenarioFile, scenario);

	// Close File
	scenarioFile.close();

	// Run simulation

	// Generate population cloud
	DebrisPopulation fragmentPopulation;

	// Generate parent debris objects
	DebrisObject primaryObject(GenerateDebrisObject(scenario["primaryObject"]));
	DebrisObject * secondaryPointer = NULL;
	// Run breakup model to generate fragment populations using settings
	if (scenario["secondaryObject"].isObject())
	{
		 DebrisObject secondaryObject(GenerateDebrisObject(scenario["secondaryObject"]));
		 secondaryPointer = &secondaryObject;
	}

	mainBreakup(fragmentPopulation, primaryObject, secondaryPointer, minLength);
	
	// Store data
	time_t dateTime = time(NULL);
	struct tm currtime;
	localtime_s(&currtime, &dateTime);
	strftime(date, sizeof(date), "%F", &currtime);

	eventType = fragmentPopulation.eventLog[0].GetEventType();

	outputFilename = "Output\\" + string(date) + "_" + eventType + ".out";
	while (fileExists(outputFilename))
	{
		ID++;
		outputFilename = "Output\\" + string(date) + "_" + eventType + '_' + to_string(ID) + ".csv";
	}
		// Create Output file
	ofstream outputFile(outputFilename, ofstream::out);
		// Write fragment data into file
	WritePopulationData(outputFile, fragmentPopulation, primaryObject, secondaryPointer);
		// Close file
	outputFile.close();
	
	// END
    return 0;
}


DebrisObject GenerateDebrisObject(Json::Value & parsedObject)
{
	double radius, mass, length, semiMajorAxis, eccentricity, inclination, rightAscension, argPerigee, meanAnomaly;
	int type;
	Json::Value elements = parsedObject["orbitalElements"];
	// Parse Json 
	radius         = parsedObject["radius"].asDouble();
	mass           = parsedObject["mass"].asDouble();
	length         = parsedObject["length"].asDouble();
	meanAnomaly    = parsedObject["meanAnomaly"].asDouble();
	type           = parsedObject["type"].asInt();
	semiMajorAxis  = elements["a"].asDouble();
	eccentricity   = elements["e"].asDouble();
	inclination    = elements["i"].asDouble();
	rightAscension = elements["Om"].asDouble();
	argPerigee     = elements["om"].asDouble();
	// Generate Object - Possible issue with reconstruction
	DebrisObject debris(radius, mass, length, semiMajorAxis, eccentricity, inclination, rightAscension, argPerigee, meanAnomaly, type);
	return debris;
}

void WritePopulationData(ofstream & dataFile, DebrisPopulation & population, DebrisObject& targetObject, DebrisObject *projectilePointer)
{
	string tempData, header, metaData, eventType;
	long ID, parentID;
	int nFrag;
	vector3D deltaV;
	double length, mass, area, areaToMass, deltaVnorm, relativeVelocity;

	// Define MetaData
	eventType = population.eventLog[0].GetEventType();
	if (eventType == "Explosion")
		metaData = "Breakup Type : " + eventType + ", Mass of Target : " + to_string(targetObject.GetMass()) + "[kg], Mass of Projectile : " + to_string(projectilePointer->GetMass()) + "[kg], Minimum Length : " +
					to_string(minLength) + "[m], Catastrophic Threshold : " + to_string(catastrophicThreshold) + "[J/g], Bridiging Function :" + bridgingFunction;
	else
	{
		relativeVelocity = (targetObject.GetVelocity() - projectilePointer->GetVelocity()).vectorNorm();
		metaData = "Breakup Type : " + eventType + ", Mass of Target : " + to_string(targetObject.GetMass()) + "[kg], Mass of Projectile : " + to_string(projectilePointer->GetMass()) + "[kg], Relative Velocity : " + to_string(relativeVelocity)
			+ "[km/s], Minimum Length : " + to_string(minLength) + "[m], Catastrophic Threshold : " + to_string(catastrophicThreshold) + "[J/g], Bridiging Function :" + bridgingFunction;
	}
	dataFile << metaData;

	// Define output format
	/* "(ParentID, ID, nFrag(representative), Length, Mass[kg], Area[m^2], A/m[m^2/kg], Dv[km/s], (dVx, dVy, dVz))" */
	header = "ParentID, ID, nFrag(representative), Length, Mass[kg], Area[m^2], A/m[m^2/kg], Dv[km/s], (dVx, dVy, dVz)";
	dataFile << header;
	// Write population data
	for (auto & debris : population.population)
	{
		// Extract Data
		parentID = debris.GetParentID();
		ID = debris.GetID();
		nFrag = debris.GetNFrag();
		length = debris.GetLength();
		mass = debris.GetMass();
		area = debris.GetArea();
		areaToMass = debris.GetAreaToMass();
		if (targetObject.GetID() == parentID)
			deltaV = debris.GetVelocity() - targetObject.GetVelocity();
		else if (projectilePointer->GetID() == parentID)
			deltaV = debris.GetVelocity() - projectilePointer->GetVelocity();
		deltaVnorm = deltaV.vectorNorm();

		// Create String
		tempData = to_string(parentID) + "," + to_string(ID) + "," + to_string(nFrag) + "," + to_string(length) + "," + to_string(mass) + "," + to_string(area) + "," + 
					to_string(areaToMass) + "," + to_string(deltaVnorm) + "," + to_string(deltaV.x) + "," + to_string(deltaV.y) + "," + to_string(deltaV.z) + "\n";

		// Pipe output
		dataFile << tempData;
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