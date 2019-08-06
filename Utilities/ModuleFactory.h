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

class ModuleFactory {
public:
	static unique_ptr<Propagator> CreatePropagator(string propagatorType, DebrisPopulation population) {
		if (propagatorType == "SGP4")
			return make_unique<SGP4>(population);
		if (propagatorType == "SimpleJ2")
			return make_unique<Analytic_J2>(population, true);
		if (propagatorType == "Simple")
			return make_unique<Analytic_J2>(population, false);
		else
			throw "Invalid Propagator Type";
	};

	static unique_ptr<CollisionAlgorithm> CreateCollisionAlgorithm(string collisionType, bool detail, double threshold) {
		if (collisionType == "Cube")
			return make_unique<CUBEApproach>(detail, threshold);
		if (collisionType == "OrbitTrace")
			return make_unique<OrbitTrace>(detail, threshold);
		if (collisionType ==  "Hoots")
			return make_unique<HootsFilter>(detail, threshold);
		else
			throw "Invalid Collision Algorithm Type";
	};

	static unique_ptr<BreakupModel> CreateBreakupModel(string breakupType, double minLength) {
		if (breakupType == "NSBM")
			return make_unique<NASABreakupModel>(minLength);
		else
			throw "Invalid Breakup Model Type";
	};
};
