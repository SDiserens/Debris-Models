// SGP4Test.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Modules\Propagators\SGP4\SGP4wrap.h"

bool fileExists(const string& name);

int main(int argc, char** argv)
{
	string arg, scenarioFilename, outputFilename, line, line2;
	vector<pair<string,string>> TLEs;
	double startTime, endTime, timeStep, minutes2days, elapsedTime;
	vector<double> stateVector;
	vector<vector<double>> stateVectorList;

	char date[100];
	int ID = 1;

	minutes2days = 1.0 / (24 * 60);

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
	cout << "Reading Scenario File : " + scenarioFilename + "...";
	ifstream scenarioFile(scenarioFilename);
	if (!scenarioFile.good())
	{
		throw std::runtime_error("Scenario file failed to load");
	}


	// Read scenario file into vector
	cout << " Parsing Scenario...";
	while (getline(scenarioFile, line))
	{
		if ((int) line[0] == 49) // 1 in asci
		{
			getline(scenarioFile, line2);
			if ((int)line2[0] == 50) // 2 in asci
				TLEs.push_back(make_pair(line, line2));
		}
	}

	// close file
	cout << " Closing Scenario File..." << endl;
	scenarioFile.close();

	
	// For each Scenario
	cout << "Propagating TLEs..." << endl;
	for (auto currentTLE : TLEs)
	{
		string scenarioDataStr = currentTLE.second.substr(70);

		// Create Population
		DebrisPopulation objectPopulation;

		// Read TLE and generate objects
			// For set in file Read 3 lines and create object
		DebrisObject object(currentTLE.first, currentTLE.second);
		object.SetInitEpoch(0.0);
		object.SetEpoch(0.0);
		objectPopulation.InitialiseEpoch(0.0);
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
		
		//objectPopulation.InitialiseEpoch(0.0);
		elapsedTime = 0.0;

		// Add identification tag
		stateVectorList.push_back(vector<double> {(double) object.GetNoradID()});

		// Store initial Position
		prop.PropagatePopulation(0.0);
		stateVector = objectPopulation.GetObject(objectID).GetStateVector();
		stateVector.insert(stateVector.begin(), elapsedTime);
		stateVectorList.push_back(stateVector);

		// Set to starting position
		if (startTime != 0.0)
		{
			// Propagate to start
			objectPopulation.GetObject(objectID).SetEpoch(startTime);
			prop.PropagatePopulation(startTime);
			elapsedTime += stod(scenarioData[0]) * minutes2days;

			// Store State vector
			stateVector = objectPopulation.GetObject(objectID).GetStateVector();
			stateVector.insert(stateVector.begin(), elapsedTime * 60 * 24);
			stateVectorList.push_back(stateVector);
		}

		// While time < endTime
		while (elapsedTime < endTime)
		{
			if (endTime - elapsedTime < timeStep)
				timeStep = endTime - elapsedTime;
			// Propagate in specified timestep
			prop.PropagatePopulation(timeStep);
			elapsedTime += timeStep;

			// Store State vector
			stateVector = objectPopulation.GetObject(objectID).GetStateVector();
			stateVector.insert(stateVector.begin(), elapsedTime * 60 * 24);
			stateVectorList.push_back(stateVector);
		}
	}

	// Store data
	time_t dateTime = time(NULL);
	struct tm currtime;
	localtime_s(&currtime, &dateTime);
	strftime(date, sizeof(date), "%F", &currtime);


	outputFilename = "Output\\" + string(date) + "_SGP4TestOutput" + ".csv";

	while (fileExists(outputFilename))
	{
		ID++;
		outputFilename = "Output\\" + string(date) + "_SGP4TestOutput" + '_' + to_string(ID) + ".csv";
	}

	cout << "Creating Data File : " + outputFilename + "...";
	// Create Output file
	ofstream outputFile;
	outputFile.open(outputFilename, ofstream::out);


	// Write state vectors for each scenario at each propagation step
	cout << "  Writing to Data File...";
	for (auto outputLine : stateVectorList)
	{
		if (outputLine.size() == 1)
		{
			outputFile << to_string(outputLine[0]) + " xx\n";
		}
		else
		{
			for (double value : outputLine)
			{
				outputFile << to_string(value) + ",";
			}
			outputFile << "\n";
		}
	}
	cout << "Finished\n";
	// Close file
	outputFile.close();

    return 0;
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
