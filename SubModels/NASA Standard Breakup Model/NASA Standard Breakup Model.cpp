// NASA Standard Breakup Model.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "../../Modules/Fragmentation_Models/NSBM.h"
#include <json\json.h>

void WritePopulationData(ofstream & dataFile, DebrisPopulation & population);
void GenerateDebrisObject(Json::Value & parsedObject, DebrisObject & debris);

int main()
{
	string scenarioFilename, outputFilename, eventType, date, line;
	float minLength;
	Json::Value config, scenario, parsedObject;
	Json::Reader reader;

	// Read config file
	ifstream configFile("config.json");
	
		// Parse config file to identify scenario file and settings
	reader.parse(configFile, config);

		// Close File
	configFile.close();

	// Read scenario file
	scenarioFilename = config["scenarioFilename"].asString();
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
	GenerateDebrisObject(scenario["objects"][0], primaryObject);
	if(scenario["objects"].size() > 1)
		GenerateDebrisObject(scenario["objects"][1], secondaryObject);

		// Run breakup model to generate fragment populations using settings
	mainBreakup(fragmentPopulation, primaryObject, &secondaryObject, minLength);

	// Store data
	outputFilename = date + "_" + eventType + ".out";
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

}

void WritePopulationData(ofstream & dataFile, DebrisPopulation & population)
{

}