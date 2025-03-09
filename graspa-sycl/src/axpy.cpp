#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "axpy.h"
#include "mc_single_particle.h"
#include "mc_swap_moves.h"

#include "print_statistics.dp.hpp"

//#include "lambda.h"
#include <numeric>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <optional>
///////////////////////////////////////////////////////////
// Wrapper for Performing a move for the selected system //
///////////////////////////////////////////////////////////
inline void RunMoves(int Cycle, Components& SystemComponents, Simulations& Sims, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, double& Rosenbluth, int SimulationMode)
{
  SystemComponents.CURRENTCYCLE = Cycle;
  //Randomly Select an Adsorbate Molecule and determine its Component: MoleculeID --> Component
  //Zhao's note: The number of atoms can be vulnerable, adding throw error here//
  if(SystemComponents.TotalNumberOfMolecules < SystemComponents.NumberOfFrameworks)
    throw std::runtime_error("There is negative number of adsorbates. Break program!");

  size_t comp = 0; // When selecting components, skip the component 0 (because it is the framework)
  size_t SelectedMolecule = 0; size_t SelectedMolInComponent = 0;

  //printf("BEFORE: Comp: %zu, TotalProb: %.5f\n", comp, SystemComponents.Moves[comp].TotalProb);
  while(SystemComponents.Moves[comp].TotalProb < 1e-10)
  {
    comp = SystemComponents.NumberOfFrameworks;
    double Random = Get_Uniform_Random();
    SelectedMolecule = (size_t) (Random*(SystemComponents.TotalNumberOfMolecules-SystemComponents.NumberOfFrameworks));
    //Zhao's note add a protection so that the selected Molecule do not exceed (or equal to) the total number of molecules//
    if(SelectedMolecule > 0 && SelectedMolecule == (SystemComponents.TotalNumberOfMolecules-SystemComponents.NumberOfFrameworks))
      SelectedMolecule --;
    SelectedMolInComponent = SelectedMolecule;
  
    //Zhao's note: Changed how molecules are selected, the previous one may cause issues in tertiary systems//
    for(size_t ijk = SystemComponents.NumberOfFrameworks; ijk < SystemComponents.Total_Components; ijk++)
    {
      if(SelectedMolInComponent == 0) break;
      if(SelectedMolInComponent >= SystemComponents.NumberOfMolecule_for_Component[ijk])
      {
        comp++;
        SelectedMolInComponent -=  SystemComponents.NumberOfMolecule_for_Component[ijk];
      }
      else
      {break;}
    }
  }

  MoveEnergy DeltaE;
  if(SystemComponents.NumberOfMolecule_for_Component[comp] == 0)
  { 
    //no molecule in the system for this species
    if(!SystemComponents.SingleSwap)
    {
      DeltaE = Insertion(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, comp);
    }
    else
    {
      DeltaE = SingleBodyMove(SystemComponents, Sims, Widom, FF, Random, SelectedMolInComponent, comp, SINGLE_INSERTION);
    }
  }
  else
  {
  double RANDOMNUMBER = Get_Uniform_Random();
  if(RANDOMNUMBER < SystemComponents.Moves[comp].TranslationProb)
  {
    //////////////////////////////
    // PERFORM TRANSLATION MOVE //
    //////////////////////////////
    DeltaE = SingleBodyMove(SystemComponents, Sims, Widom, FF, Random, SelectedMolInComponent, comp, TRANSLATION);
  }
  else if(RANDOMNUMBER < SystemComponents.Moves[comp].RotationProb) //Rotation
  {
    ///////////////////////////
    // PERFORM ROTATION MOVE //
    ///////////////////////////
    DeltaE = SingleBodyMove(SystemComponents, Sims, Widom, FF, Random, SelectedMolInComponent, comp, ROTATION);
  }
  else if(RANDOMNUMBER < SystemComponents.Moves[comp].ReinsertionProb)
  {
    //////////////////////////////
    // PERFORM REINSERTION MOVE //
    //////////////////////////////
    DeltaE = Reinsertion(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, comp);
  }
  else if(RANDOMNUMBER < SystemComponents.Moves[comp].SwapProb)
  {
    ////////////////////////////
    // PERFORM GCMC INSERTION //
    ////////////////////////////
    if(Get_Uniform_Random() < 0.5)
    {
      if(!SystemComponents.SingleSwap)
      {
        DeltaE = Insertion(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, comp);
      }
      else
      {
        DeltaE = SingleBodyMove(SystemComponents, Sims, Widom, FF, Random, SelectedMolInComponent, comp, SINGLE_INSERTION);
        //DeltaE = SingleSwapMove(SystemComponents, Sims, Widom, FF, Random, SelectedMolInComponent, comp, SINGLE_INSERTION);
      }
    }
    else
    {
      ///////////////////////////
      // PERFORM GCMC DELETION //
      ///////////////////////////
      //Zhao's note: Do not do a deletion if the chosen molecule is a fractional molecule, fractional molecules should go to CBCFSwap moves//
      if(!((SystemComponents.hasfractionalMolecule[comp]) && SelectedMolInComponent == SystemComponents.Lambda[comp].FractionalMoleculeID))
      {
        if(!SystemComponents.SingleSwap)
        {
          DeltaE = Deletion(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, comp);
        }
        else
        {
          DeltaE = SingleBodyMove(SystemComponents, Sims, Widom, FF, Random, SelectedMolInComponent, comp, SINGLE_DELETION);
        }
      }
    }
  }
  }
  /*
  if(Cycle == 6)
  {
    printf("Cycle [%d], Printing DeltaE\n", Cycle);
    DeltaE.print();
  }
  */
  SystemComponents.deltaE += DeltaE;
}

double Run_Simulation_ForOneBox(int Cycles, Components& SystemComponents, Simulations& Sims, ForceField FF, RandomNumber& Random, WidomStruct Widom, double init_energy, int SimulationMode, bool SetMaxStep, size_t MaxStepPerCycle, Units Constants)
{
  std::vector<size_t>CBCFPerformed(SystemComponents.Total_Components);
  size_t WLSampled = 0; size_t WLAdjusted = 0;

  int BlockAverageSize = 1;
  if(SimulationMode == PRODUCTION)
  {
    BlockAverageSize = Cycles / SystemComponents.Nblock;
    if(Cycles % SystemComponents.Nblock != 0)
      printf("Warning! Number of Cycles cannot be divided by Number of blocks. Residue values go to the last block\n");
  }
 
  if(SimulationMode == EQUILIBRATION) //Rezero the TMMC stats at the beginning of the Equilibration cycles//
  {
    //Clear TMMC data in the collection matrix//
    for(size_t comp = 0; comp < SystemComponents.Total_Components; comp++)
      SystemComponents.Tmmc[comp].ClearCMatrix();
  }
  //Clear Rosenbluth weight statistics after Initialization//
  if(SimulationMode == EQUILIBRATION)
  {
    for(size_t comp = 0; comp < SystemComponents.Total_Components; comp++)
      for(size_t i = 0; i < SystemComponents.Nblock; i++)
        SystemComponents.Moves[comp].ClearRosen(i);
  }
  double running_energy = 0.0;
  double running_Rosenbluth = 0.0;

  /*
  /////////////////////////////////////////////
  // FINALIZE (PRODUCTION) CBCF BIASING TERM //
  /////////////////////////////////////////////
  if(SimulationMode == PRODUCTION)
  {
    for(size_t icomp = 0; icomp < SystemComponents.Total_Components; icomp++)
      if(SystemComponents.hasfractionalMolecule[icomp])
        Finalize_WangLandauIteration(SystemComponents.Lambda[icomp]);
  }

  ///////////////////////////////////////////////////////////////////////
  // FORCE INITIALIZING CBCF BIASING TERM BEFORE INITIALIZATION CYCLES //
  ///////////////////////////////////////////////////////////////////////
  if(SimulationMode == INITIALIZATION && Cycles > 0)
  {
    for(size_t icomp = 0; icomp < SystemComponents.Total_Components; icomp++)
      if(SystemComponents.hasfractionalMolecule[icomp])
        Initialize_WangLandauIteration(SystemComponents.Lambda[icomp]);
  }
  */

  for(size_t i = 0; i < Cycles; i++)
  {
    size_t Steps = 20;
    if(Steps < SystemComponents.TotalNumberOfMolecules)
    {
      Steps = SystemComponents.TotalNumberOfMolecules;
    }
    //Determine BlockID//
    for(size_t comp = 0; comp < SystemComponents.Total_Components; comp++){
      BlockAverageSize = Cycles / SystemComponents.Nblock;
      if(BlockAverageSize > 0) SystemComponents.Moves[comp].BlockID = i/BlockAverageSize; 
      if(SystemComponents.Moves[comp].BlockID >= SystemComponents.Nblock) SystemComponents.Moves[comp].BlockID--;   }
    ////////////////////////////////////////
    // Zhao's note: for debugging purpose //
    ////////////////////////////////////////
    if(SetMaxStep && Steps > MaxStepPerCycle) Steps = MaxStepPerCycle;
    for(size_t j = 0; j < Steps; j++)
    {
      RunMoves(i, SystemComponents, Sims, FF, Random, Widom, running_Rosenbluth, SimulationMode);
    }
    /*
    //////////////////////////////////////////////
    // SAMPLE (EQUILIBRATION) CBCF BIASING TERM //
    //////////////////////////////////////////////
    if(SimulationMode == EQUILIBRATION && i%50==0)
    {
      for(size_t icomp = 0; icomp < SystemComponents.Total_Components; icomp++)
      { //Try to sample it if there are more CBCF moves performed//
        if(SystemComponents.hasfractionalMolecule[icomp])
        {
          Sample_WangLandauIteration(SystemComponents.Lambda[icomp]);
          CBCFPerformed[icomp] = SystemComponents.Moves[icomp].CBCFTotal; WLSampled++;
        }
      }
    }
    */
    if(i%500==0)
    {
      for(size_t comp = 0; comp < SystemComponents.Total_Components; comp++)
      {  
        if(SystemComponents.Moves[comp].TranslationTotal > 0)
          Update_Max_Translation(SystemComponents.Moves[comp], Sims);
        if(SystemComponents.Moves[comp].RotationTotal > 0)
          Update_Max_Rotation(SystemComponents.Moves[comp], Sims);
      }
    }
    if(i%5000==0) Print_Cycle_Statistics(i, SystemComponents);
    /*
    ////////////////////////////////////////////////
    // ADJUST CBCF BIASING FACTOR (EQUILIBRATION) //
    ////////////////////////////////////////////////
    if(i%5000==0 && SimulationMode == EQUILIBRATION)
    {
      for(size_t icomp = 0; icomp < SystemComponents.Total_Components; icomp++)
        if(SystemComponents.hasfractionalMolecule[icomp])
        {  Adjust_WangLandauIteration(SystemComponents.Lambda[icomp]); WLAdjusted++;}
    }
    */
    if(SimulationMode == PRODUCTION)
    {
      //Record values for energy//
      Gather_Averages_Types(SystemComponents.EnergyAverage, init_energy, running_energy, i, BlockAverageSize, SystemComponents.Nblock);
      //Record values for Number of atoms//
      for(size_t comp = 0; comp < SystemComponents.Total_Components; comp++)
        Gather_Averages_Types(SystemComponents.Moves[comp].MolAverage, SystemComponents.NumberOfMolecule_for_Component[comp], 0.0, i, BlockAverageSize, SystemComponents.Nblock);
    }
    if(SimulationMode != INITIALIZATION && i > 0)
    {
      for(size_t comp = 0; comp < SystemComponents.Total_Components; comp++)
        if(i % SystemComponents.Tmmc[comp].UpdateTMEvery == 0)
          SystemComponents.Tmmc[comp].AdjustTMBias();
    }
  }
  //print statistics
  if(Cycles > 0)
  {
    if(SimulationMode == EQUILIBRATION) printf("Sampled %zu WangLandau, Adjusted WL %zu times\n", WLSampled, WLAdjusted);
    PrintAllStatistics(SystemComponents, Sims, Cycles, SimulationMode, BlockAverageSize);
    //Print_Widom_Statistics(SystemComponents, Sims.Box, Constants, 1);
  }
  return running_energy;
}
