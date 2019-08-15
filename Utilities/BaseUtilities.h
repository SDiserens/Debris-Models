#pragma once

bool fileExists(const string& name);

DebrisObject GenerateDebrisObject(Json::Value & parsedObject);

void LoadScenario(DebrisPopulation & population, string scenarioFilename);

void WriteCollisionData(string scenario, Json::Value & config, string collisionModel, Json::Value & collisionConfig, vector<tuple<double, pair<long, long>, double>> collisionLog);