#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <complex>
#include "VDW_Coulomb.dp.hpp"
//#include "RN.h"
#include <vector>
//#include "read_data.h"
double Run_Simulation(int Cycles, Components& SystemComponents, Simulations& Sims, ForceField FF, RandomNumber Random, WidomStruct Widom, double init_energy, std::vector<size_t>& NumberOfCreateMolecules, int SimulationMode, bool AlreadyHasFractionalMolecule = false);

double Run_Simulation_ForOneBox(int Cycles, Components& SystemComponents, Simulations& Sims, ForceField FF, RandomNumber& Random, WidomStruct Widom, double init_energy, int SimulationMode, bool SetMaxStep, size_t MaxStepPerCycle, Units Constants);
