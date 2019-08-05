#pragma once


DebrisObject GenerateDebrisObject(Json::Value & parsedObject);

void LoadScenario(DebrisPopulation & population, string scenarioFilename);