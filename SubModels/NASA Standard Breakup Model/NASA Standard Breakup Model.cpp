// NASA Standard Breakup Model.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "../../Modules/Fragmentation_Models/NSBM.h"


void WritePopulationData(ofstream & dataFile, DebrisPopulation & population, Event breakupEvent, Json::Value & config);

int main()
{
	string scenarioFilename, outputFilename, eventType, bridgingFunction;
	char date[100];
	int numFragBuckets, ID = 1;
	double minLength, catastrophicThreshold, scaling;

	Json::Value config, scenario, parsedObject;
	Json::Reader reader;

	// ----------------------------
	// - Parsing config variables -
	// ----------------------------
	LoadConfigFile(config);

	minLength = config["minLength"].asDouble();
	scenarioFilename = "Scenarios\\" + config["scenarioFilename"].asString();
	numFragBuckets = config["numberOfBuckets"].asInt();
	bridgingFunction = config["bridgingFunction"].asString();
	catastrophicThreshold = config["catastrophicThreshold"].asDouble();
	scaling = config["scaling"].asDouble();

	// Create Breakup Model
	NASABreakupModel NSBM(minLength, catastrophicThreshold, numFragBuckets, bridgingFunction, scaling);


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
	// Close File

	cout << " Closing Scenario File...\n";
	scenarioFile.close();

	// Run simulation

	cout << "Preparing Objects for Breakup...";
	// Generate population cloud
	DebrisPopulation fragmentPopulation;
	Event breakupEvent;

	// Generate parent debris objects
	DebrisObject primaryObject(GenerateDebrisObject(scenario["primaryObject"]));
	fragmentPopulation.AddDebrisObject(primaryObject);
	// Run breakup model to generate fragment populations using settings
	if (scenario["secondaryObject"].isObject())
	{
		 DebrisObject secondaryObject(GenerateDebrisObject(scenario["secondaryObject"]));
		 fragmentPopulation.AddDebrisObject(secondaryObject);
		 double relativeV = (primaryObject.GetVelocity() - secondaryObject.GetVelocity()).vectorNorm();
		 breakupEvent = Event(0. , primaryObject.GetID(), secondaryObject.GetID(), relativeV, primaryObject.GetMass() +  secondaryObject.GetMass(), primaryObject.GetElements().GetRadialPosition());
	}
	else
		breakupEvent = Event(0, primaryObject.GetID(), primaryObject.GetMass());

	cout << "    Simulating Breakup...";
	NSBM.mainBreakup(fragmentPopulation, breakupEvent);
	cout << "    Breakup Simulation Complete\n";

	// Store data
	time_t dateTime = time(NULL);
	struct tm currtime;
	localtime_s(&currtime, &dateTime);
	strftime(date, sizeof(date), "%F", &currtime);

	eventType = fragmentPopulation.GetEventLog()[0].GetEventTypeString();

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
		// Write fragment data into file
	cout << "  Writing to Data File...";
	WritePopulationData(outputFile, fragmentPopulation, breakupEvent, config);
	cout << "Finished\n";
		// Close file
	outputFile.close();
	
	// END
    return 0;
}


void WritePopulationData(ofstream & dataFile, DebrisPopulation & population, Event breakupEvent, Json::Value & config)
{
	string tempData, header, metaData, eventType;
	long ID, parentID;
	int nFrag;
	vector3D deltaV;
	double length, mass, area, areaToMass, deltaVnorm, relativeVelocity;
	DebrisObject targetObject = population.GetObject(breakupEvent.GetPrimary());
	DebrisObject projectileObject;

	// Define MetaData
	eventType = breakupEvent.GetEventTypeString();
	string bridgingFunction = config["bridgingFunction"].asString();
	double catastrophicThreshold = config["catastrophicThreshold"].asDouble();
	double minLength = config["minLength"].asDouble();

	if (eventType == "Explosion")
		metaData = "Breakup Type : ," + eventType + "\n Mass of Target :," + to_string(targetObject.GetMass()) + ",[kg]\n Mass of Projectile : ," + "N/A" + ",[kg]\n Minimum Length : ," +
					to_string(minLength) + ",[m]\n Catastrophic Threshold : ," + to_string(catastrophicThreshold) + ",[J/g]\n Bridiging Function :," + bridgingFunction;
	else
	{
		projectileObject = population.GetObject(breakupEvent.GetSecondary());
		relativeVelocity = breakupEvent.relativeVelocity;
		metaData = "Breakup Type : ," + eventType + "\n Mass of Target : ," + to_string(targetObject.GetMass()) + ",[kg]\n Mass of Projectile : ," + to_string(projectileObject.GetMass()) + ",[kg]\n Relative Velocity : ," + to_string(relativeVelocity)
			+ ",[km/s]\n Minimum Length : ," + to_string(minLength) + ",[m]\n Catastrophic Threshold : ," + to_string(catastrophicThreshold) + ",[J/g]\n Bridiging Function :," + bridgingFunction;
	}
	dataFile << metaData + "\n";

	// Define output format
	/* "(ParentID, ID, nFrag(representative), Length, Mass[kg], Area[m^2], A/m[m^2/kg], Dv[km/s], (dVx, dVy, dVz))" */
	header = "ParentID, ID, nFrag(representative), Length, Mass[kg], Area[m^2], A/m[m^2/kg], Dv[km/s], (dVx, dVy, dVz)";
	dataFile << header + "\n";
	// Write population data
	DebrisObject debris;
	for (auto const& debrisID : population.population)
	{
		debris = DebrisObject(debrisID.second);
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
		else if (projectileObject.GetID() == parentID)
			deltaV = debris.GetVelocity() - projectileObject.GetVelocity();
		deltaVnorm = deltaV.vectorNorm();

		// Create String
		tempData = to_string(parentID) + "," + to_string(ID) + "," + to_string(nFrag) + "," + to_string(length) + "," + to_string(mass) + "," + to_string(area) + "," +
			to_string(areaToMass) + "," + to_string(deltaVnorm) + "," + to_string(deltaV.x) + "," + to_string(deltaV.y) + "," + to_string(deltaV.z);

		// Pipe output
		dataFile << tempData + "\n";
	}
}
