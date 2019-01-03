// NASA Standard Breakup Model.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"


int main()
{
	string scenarioFilename, outputFilename, eventType, date, line;

	// Read config file
	ifstream configFile("config.txt");

		// Parse config file to identify scenario file and settings

		// Close File
	configFile.close();

	// Read scenario file
	ifstream scenarioFile(scenarioFilename);

		// Parse scenario file to identify object characteristics

		// Close File
	scenarioFile.close();

	// Run simulation

		// Generate population cloud

		// Generate parent debris objects

		// Run breakup model to generate fragment populations using settings

	// Store data
	outputFilename = date + "_" + eventType + ".out";
		// Create Output file
	ofstream outputFile(outputFilename, ofstream::out);
		// Write fragment data into file

		// Close file
	outputFile.close();

	// END
    return 0;
}

