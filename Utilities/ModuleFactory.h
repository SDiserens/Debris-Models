#pragma once
//
//
//enum FragmentationModules {
//	NSBM
//};
//
//enum CollisionModules {
//	Cube,
//	OrbitTrace,
//	Hoots
//};
//
//enum PropagationModules {
//	SGP,
//	SimpleJ2
//};

#include "stdafx.h"

class ModuleFactory {
public:
	static unique_ptr<Propagator> CreatePropagator(string propagatorType, DebrisPopulation & population, Json::Value & config) {
		if (propagatorType == "SGP4")
			return CreateSGP4Instance(population, config);
		if (propagatorType == "SimpleJ2")
			return make_unique<Analytic_J2>(population, true);
		if (propagatorType == "Simple")
			return make_unique<Analytic_J2>(population, false);
		else
			throw "Invalid Propagator Type";
	};

	static unique_ptr<CollisionAlgorithm> CreateCollisionAlgorithm(string collisionType, Json::Value & config) {
		if (collisionType == "Cube")
			return CreateCubeInstance(config, false);
		if (collisionType == "Cube-offset")
			return CreateCubeInstance(config, true);
		if (collisionType == "OrbitTrace")
			return CreateOTInstance(config);
		if (collisionType ==  "Hoots")
			return CreateHootsInstance(config);
		else
			throw "Invalid Collision Algorithm Type";
	};

	static void  UpdateCollisionThreshold(string collisionType, Json::Value & config, double threshold) {
		if (collisionType == "Cube" || collisionType == "Cube-offset")
			config["CubeDimension"] = threshold;
		else if (collisionType == "OrbitTrace")
			config["ConjunctionThreshold"] = threshold;
		else if (collisionType == "Hoots")
			config["ConjunctionThreshold"] = threshold;
		else
			throw "Invalid Collision Algorithm Type";
	}

	static unique_ptr<BreakupModel> CreateBreakupModel(string breakupType, Json::Value & config) {
		if (breakupType == "NSBM")
			return CreateNSBMInstance(config);
		else if (breakupType == "NSBM-new")
			return CreateNSBMNewInstance(config);
		else if (breakupType.empty())
			return nullptr;
		else
			throw "Invalid Breakup Model Type";
	};


private:
	// Propagator Factories
	static unique_ptr<SGP4> CreateSGP4Instance(DebrisPopulation & population, Json::Value & config){
		gravconsttype gravModel;
		char opsMode = config["opsMode"].asString()[0];
		string gravType = config["gravModel"].asString();

		if (gravType == "wgs72")
			gravModel = wgs72;
		else if (gravType == "wgs72old")
			gravModel = wgs72old;
		else if (gravType == "wgs84")
			gravModel = wgs84;
		else
			throw "Invalid Gravity Model Type";

		return make_unique<SGP4>(population, opsMode, gravModel);
	};

	// Collision Factories
	static unique_ptr<CUBEApproach> CreateCubeInstance(Json::Value & config, bool offsetCubes) {
		// Read Cube config
		bool probabilities = config["Verbose"].asBool();
		double dimension = config["CubeDimension"].asDouble();
		int mcRuns = config["CubeMC"].asInt();
		bool offset = offsetCubes;

		return make_unique<CUBEApproach>(probabilities, dimension, mcRuns, offset);
	};

	static unique_ptr<OrbitTrace> CreateOTInstance(Json::Value & config) {
		// Read OT config
		bool probabilities = config["Verbose"].asBool();
		double threshold = config["ConjunctionThreshold"].asDouble();
		int MOIDtype = config["MOIDtype"].asInt();

		return make_unique<OrbitTrace>( probabilities, threshold, MOIDtype);
	};

	static unique_ptr<HootsFilter> CreateHootsInstance(Json::Value & config) {
		// Read Hoots config
		bool times = config["Verbose"].asBool();
		double conjunctionThreshold = config["ConjunctionThreshold"].asDouble();
		double collisionThreshold = config["CollisionThreshold"].asDouble();

		return make_unique<HootsFilter>( times, conjunctionThreshold, collisionThreshold);
	};

	// Fragmentation Factories
	static unique_ptr<NASABreakupModel> CreateNSBMInstance(Json::Value & config) {
		// Read NSBM config
		double minLength = config["minLength"].asDouble();
		double catastrophicThreshold = config["catastrophicThreshold"].asDouble();
		int numberBuckets = config["numberBuckets"].asInt();
		string bridgingFucntion = config["bridgingFucntion"].asString();
		double explosionScaling = config["explosionScaling"].asDouble();
		double representativeFragmentThreshold = config["representativeFragmentThreshold"].asDouble();
		int representativeFragmentNumber = config["representativeFragmentNumber"].asInt();

		return make_unique<NASABreakupModel>(minLength, catastrophicThreshold, numberBuckets, bridgingFucntion, explosionScaling, representativeFragmentThreshold, representativeFragmentNumber);
	};	

	static unique_ptr<NASABreakupModel> CreateNSBMNewInstance(Json::Value & config) {
		// Read NSBM config
		double minLength = config["minLength"].asDouble();
		double catastrophicThreshold = config["catastrophicThreshold"].asDouble();
		int numberBuckets = config["numberBuckets"].asInt();
		string bridgingFucntion = config["bridgingFucntion"].asString();
		double explosionScaling = config["explosionScaling"].asDouble();
		double representativeFragmentThreshold = config["representativeFragmentThreshold"].asDouble();
		int representativeFragmentNumber = config["representativeFragmentNumber"].asInt();

		unique_ptr<NASABreakupModel> model = make_unique<NASABreakupModel>(minLength, catastrophicThreshold, numberBuckets, bridgingFucntion, explosionScaling, representativeFragmentThreshold, representativeFragmentNumber);
		model->SetNewSpaceParameters();
		return model;
	};
};
