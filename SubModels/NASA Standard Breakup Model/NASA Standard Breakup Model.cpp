// NASA Standard Breakup Model.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "../../Modules/Fragmentation_Models/NSBM.h"
#include <json\json.h>

void WritePopulationData(ofstream & dataFile, DebrisPopulation & population);
void GenerateDebrisObject(Json::Value & parsedObject, DebrisObject & debris);
bool fileExists(const string& name);

int main()
{
	string scenarioFilename, outputFilename, eventType, line;
	char date[10];
	int ID = 1;
	double minLength;
	Json::Value config, scenario, parsedObject;
	Json::Reader reader;

	// Read config file
	ifstream configFile("config.json");
	
	// Parse config file to identify scenario file and settings
	reader.parse(configFile, config);

	minLength = config["minLength"].asDouble();
	scenarioFilename = config["scenarioFilename"].asString();
	numFragBuckets = config["numberOfBuckets"].asInt();
	bridgingFunction = config["bridgingFunction"].asString();

	// Close File
	configFile.close();

	// Read scenario file
	ifstream scenarioFile(scenarioFilename);

	// Parse scenario file to identify object characteristics
	reader.parse(scenarioFile, scenario);

	// Close File
	scenarioFile.close();

	// Run simulation

		// Generate population cloud
	DebrisPopulation fragmentPopulation;
		// Generate parent debris objects
	DebrisObject primaryObject, secondaryObject;
	GenerateDebrisObject(scenario["primaryObject"], primaryObject);
	if (scenario["secondaryObject"].isObject())
		GenerateDebrisObject(scenario["secondaryObject"], secondaryObject);

		// Run breakup model to generate fragment populations using settings
	mainBreakup(fragmentPopulation, primaryObject, &secondaryObject, minLength);

	// Store data
	time_t dateTime;
	strftime(date, sizeof(date), "%F", localtime(&dateTime));

	eventType = fragmentPopulation.eventLog[0].GetEventType();

	outputFilename = "Output\\" + string(date) + "_" + eventType + ".out";
	while (fileExists(outputFilename))
	{
		ID++;
		outputFilename = "Output\\" + string(date) + "_" + eventType + '_' + to_string(ID) + ".out";
	}
		// Create Output file
	ofstream outputFile(outputFilename, ofstream::out);
		// Write fragment data into file
	WritePopulationData(outputFile, fragmentPopulation);
		// Close file
	outputFile.close();
	
	// END
    return 0;
}


void GenerateDebrisObject(Json::Value & parsedObject, DebrisObject & debris)
{
	double radius, mass, length, semiMajorAxis, eccentricity, inclination, rightAscension, argPerigee, meanAnomaly;

	// Parse Json 
	radius         = parsedObject["radius"].asDouble();
	mass           = parsedObject["mass"].asDouble();
	length         = parsedObject["length"].asDouble();
	semiMajorAxis  = parsedObject["orbitalElements"]["a"].asDouble();
	eccentricity   = parsedObject["orbitalElements"]["e"].asDouble();
	inclination    = parsedObject["orbitalElements"]["i"].asDouble();
	rightAscension = parsedObject["orbitalElements"]["Om"].asDouble();
	argPerigee     = parsedObject["orbitalElements"]["om"].asDouble();
	meanAnomaly    = parsedObject["meanAnomaly"].asDouble();

	// Generate Object - Possible issue with reconstruction
	debris = DebrisObject(radius, mass, length, semiMajorAxis, eccentricity, inclination, rightAscension, argPerigee, meanAnomaly);
}

void WritePopulationData(ofstream & dataFile, DebrisPopulation & population)
{
	//TODO - Define output format
	/* (ParentID, ID, nFrag(representative), Length, Mass[kg], Area[m^2], A/m[m^2/kg], Dv[km/s], (dVx, dVy, dVz) */

	// TODO - Write population data
}

bool fileExists(const string& name)
{
	if (FILE *file = fopen(name.c_str(), "r")) 
	{
		fclose(file);
		return true;
	}
	else 
		return false;
}