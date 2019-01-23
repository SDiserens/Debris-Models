// CUBE.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Modules/Collision_Algorithms/CUBE.h"
#include <json\json.h>


#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <ctime>


void WriteCollisionData(ofstream & dataFile);
DebrisObject GenerateDebrisObject(Json::Value & parsedObject);
bool fileExists(const string& name);

int main()
{
	string scenarioFilename, outputFilename, eventType;
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
	// TODO - Identify config variables

	cout << " Closing Config File...\n";
	// Close File
	configFile.close();

	cout << "Reading Scenario File : " + scenarioFilename + "...";
	// Read scenario file
	ifstream scenarioFile(scenarioFilename);
	if (!scenarioFile.good())
	{
		throw std::runtime_error("Scenario file failed to load");
	}

	// Parse scenario file to identify object characteristics
	reader.parse(scenarioFile, scenario);

	cout << " Parsing Scenario...";
	// TODO - Create population of objects

	// Close File
	cout << " Closing Scenario File...\n";
	scenarioFile.close();

	// Run simulation
	//TODO - Call CUBE approach

	// Store data
	time_t dateTime = time(NULL);
	struct tm currtime;
	localtime_s(&currtime, &dateTime);
	strftime(date, sizeof(date), "%F", &currtime);

	eventType = "Test";

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
	WriteCollisionData(outputFile);
	cout << "Finished\n";
	// Close file
	outputFile.close();

	// END
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

void WriteCollisionData(ofstream & dataFile)
{
	// Determine output format
	/* 
	MetaData = [['Simulation:', 'Jovian_Moons'], ['Cube Dimension', Dim, 'km'], ['Number of sections, N:', N], ['Section length, n:', n, 'days']]

	Collision Pairs :-
	Total Collision Rate per pair :-
	Total Conjunction Rate per pair :-
	----
	Rate per evaluation step for each pair
	----
	*/

	// TODO - Write data to file
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