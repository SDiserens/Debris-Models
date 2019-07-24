// SGP4Test.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Modules\Propagators\SGP4\SGP4wrap.h"


int main(int argc, char** argv)
{
	string arg, scenarioFilename, outputFilename, line, line2;
	vector<pair<string,string>> TLEs;
	double startTime, endTime, timeStep, minutes2days, elapsedTime;
	vector<double> stateVector;
	vector<vector<double>> stateVectorList;

	minutes2days = 1 / (24 * 60);

	scenarioFilename = "SGP4-VER.TLE";
	// Parse command line arguments
	for (int i = 1; i < argc; ++i) {
		arg = argv[i];
		if ((arg == "-f") || (arg == "--filename"))
		{
			scenarioFilename = argv[++i];
		}
	}
	
	// Open scenario file
	ifstream scenarioFile(scenarioFilename);
	if (!scenarioFile.good())
	{
		throw std::runtime_error("Scenario file failed to load");
	}


	// Read scenario file into vector
	while (getline(scenarioFile, line))
	{
		if (line[0] == 1)
		{
			getline(scenarioFile, line2);
			if (line2[0] == 2)
				TLEs.push_back(make_pair(line, line2));
		}
	}

	// close file
	scenarioFile.close();

	
	// For each Scenario
	for (auto currentTLE : TLEs)
	{
		string scenarioDataStr = currentTLE.second.substr(70);

		// Create Population
		DebrisPopulation objectPopulation;

		// Read TLE and generate objects
			// For set in file Read 3 lines and create object
		DebrisObject object(currentTLE.first, currentTLE.second);
			// Add to Population
		objectPopulation.AddDebrisObject(object);
		long objectID = object.GetID();

		SGP4 prop(objectPopulation);

		//Set Start time
		istringstream iss(scenarioDataStr);
		vector<string> scenarioData(istream_iterator<string>{iss}, istream_iterator<string>());

		startTime = stod(scenarioData[0]) * minutes2days;
		endTime = stod(scenarioData[1]) * minutes2days;
		timeStep = stod(scenarioData[2]) * minutes2days;

		objectPopulation.InitialiseEpoch(startTime);
		elapsedTime = 0.0;

		stateVector = objectPopulation.GetObject(objectID).GetStateVector();
		stateVector.insert(stateVector.begin(), elapsedTime);
		stateVectorList.push_back(stateVector);

		if (startTime != 0.0)
		{
			prop.PropagatePopulation(startTime);
			elapsedTime += startTime;

			stateVector = objectPopulation.GetObject(objectID).GetStateVector();
			stateVector.insert(stateVector.begin(), elapsedTime);
			stateVectorList.push_back(stateVector);
		}
		// While time < endTime
		while (objectPopulation.GetEpoch() < endTime)
		{
			// Propagate in specified timestep
			prop.PropagatePopulation(timeStep);
			elapsedTime += stod(scenarioData[2]);

			// Store State vector
			stateVector = objectPopulation.GetObject(objectID).GetStateVector();
			stateVector.insert(stateVector.begin(), elapsedTime);
			stateVectorList.push_back(stateVector);
		}
	}

	// Create Output file
		// Write state vectors for each scenario at each propagation step

    return 0;
}

