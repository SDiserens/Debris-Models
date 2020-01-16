#pragma once

bool fileExists(const string& name);

DebrisObject GenerateDebrisObject(Json::Value & parsedObject, double epoch=0);

void LoadScenario(DebrisPopulation & population, string scenarioFilename);

void WriteCollisionData(string scenario, Json::Value & config, string collisionModel, Json::Value & collisionConfig, vector<tuple<int, double, pair<string, string>, double, double>> collisionLog);

void WriteSimulationData(string scenario, Json::Value & config, string collisionModel, Json::Value & collisionConfig, string propagatorType,
						 Json::Value & propagatorConfig, string breakUpType, Json::Value & fragmentationConfig, vector<tuple<int, double, int, tuple<int, int, int>, int, tuple<int, int, int>>> simulationLog);

void WriteEventData(string scenario, Json::Value & config, string collisionModel, Json::Value & collisionConfig, string propagatorType,
					Json::Value & propagatorConfig, string breakUpType, Json::Value & fragmentationConfig, vector<Event> eventLog);