#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "mc_utilities.h"
#include <cmath>

void get_nsim(Simulations* Sims, size_t *nsim, size_t *blockbefore, size_t NumberOfSimulations, size_t BlockID)
{
  size_t totalblock = 0; size_t tempsim = 0;
  for(size_t i = 0; i < NumberOfSimulations; i++)
  {
    if(BlockID >= (totalblock + Sims[i].Nblocks))
    {
      tempsim++;
      totalblock += Sims[i].Nblocks;
    }
  }
  *nsim        = tempsim;
  *blockbefore = totalblock;
}

void get_position(const Atoms* System, size_t *posi, size_t *comp, size_t i, size_t NumberComp)
{
  size_t temppos = i; size_t totalsize = 0; size_t tempcomp = 0;
  for(size_t ijk = 0; ijk < NumberComp; ijk++)
  {
    totalsize += System[ijk].size;
    if(temppos >= totalsize)
    {
      tempcomp++;
      temppos -= System[ijk].size;
    }
  }
  *posi = temppos;
  *comp = tempcomp;
}

////////////////////////////////////////////////
// Generalized function for single Body moves //
////////////////////////////////////////////////
static inline MoveEnergy SingleBodyMove(Components& SystemComponents, Simulations& Sims, WidomStruct& Widom, ForceField& FF, RandomNumber& Random, size_t SelectedMolInComponent, size_t SelectedComponent, int MoveType)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  //Get Number of Molecules for this component (For updating TMMC)//
  double NMol = SystemComponents.NumberOfMolecule_for_Component[SelectedComponent];
  if(SystemComponents.hasfractionalMolecule[SelectedComponent]) NMol--;

  bool    Do_New = false;
  bool    Do_Old = false;
  double3 Max = {0.0, 0.0, 0.0};

  size_t Atomsize = 0;
  for(size_t ijk = 0; ijk < SystemComponents.Total_Components; ijk++)
    Atomsize += SystemComponents.Moleculesize[ijk] * SystemComponents.NumberOfMolecule_for_Component[ijk];
  size_t Molsize = SystemComponents.Moleculesize[SelectedComponent]; //Get the size of the selected Molecule
  //Set up Old position and New position arrays
  if(Molsize >= 1024)
  {
    throw std::runtime_error("Molecule size is greater than allocated size, Why so big?\n");
  }
  size_t start_position = SelectedMolInComponent*SystemComponents.Moleculesize[SelectedComponent];

  switch (MoveType)
  {
    case TRANSLATION:
    {
      SystemComponents.Moves[SelectedComponent].TranslationTotal++;
      Do_New = true; Do_Old = true; 
      Max = Sims.MaxTranslation;
      break;
    }
    case ROTATION:
    {
      SystemComponents.Moves[SelectedComponent].RotationTotal++;
      Do_New = true; Do_Old = true;
      Max = Sims.MaxRotation;
      break;
    }
    case SINGLE_INSERTION:
    {
      SystemComponents.Moves[SelectedComponent].InsertionTotal++;
      Do_New = true;
      start_position = 0;
      break;
    } 
    case SINGLE_DELETION:
    {
      SystemComponents.Moves[SelectedComponent].DeletionTotal++;
      Do_Old = true;
      break;
    }
  }
  if(!Do_New && !Do_Old) throw std::runtime_error("Doing Nothing For Single Particle Move?\n");

  //Zhao's note: possible bug, you may only need 3 instead of 3 * N random numbers//
  Random.Check(Molsize);
  /*
  DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  /*
  DPCT1069:67: The argument 'Sims' of the kernel function contains virtual
  pointer(s), which cannot be dereferenced. Try to migrate the code with
  "usm-level=restricted".
  */
  /*
  DPCT1069:68: The argument 'FF' of the kernel function contains virtual
  pointer(s), which cannot be dereferenced. Try to migrate the code with
  "usm-level=restricted".
  */
  q_ct1.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, Molsize),
                                       sycl::range<3>(1, 1, Molsize)),
                     [=](sycl::nd_item<3> item_ct1) {
                       get_new_position(Sims, FF, start_position,
                                        SelectedComponent, Random.device_random, Max,
                                        Random.offset, MoveType, item_ct1);
                     }).wait();
  Random.Update(Molsize);

  // Setup for the pairwise calculation //
  // New Features: divide the Blocks into two parts: Host-Guest + Guest-Guest //
  size_t NHostAtom = 0; size_t NGuestAtom = 0;
  for (size_t i = 0; i < SystemComponents.NComponents.y(); i++)
    NHostAtom += SystemComponents.Moleculesize[i] * SystemComponents.NumberOfMolecule_for_Component[i];
  for (size_t i = SystemComponents.NComponents.y();
       i < SystemComponents.NComponents.x(); i++)
    NGuestAtom+= SystemComponents.Moleculesize[i] * SystemComponents.NumberOfMolecule_for_Component[i];

  size_t HG_Nthread=0; size_t HG_Nblock=0; Setup_threadblock(NHostAtom *  Molsize, &HG_Nblock, &HG_Nthread);
  size_t GG_Nthread=0; size_t GG_Nblock=0; Setup_threadblock(NGuestAtom * Molsize, &GG_Nblock, &GG_Nthread);
  size_t HGGG_Nthread = std::max(HG_Nthread, GG_Nthread);
  size_t HGGG_Nblock  = HG_Nblock + GG_Nblock;
  //printf("Total_Comp: %zu, Host Comp: %zu, Adsorbate Comp: %zu\n", SystemComponents.NComponents.x, SystemComponents.NComponents.y, SystemComponents.NComponents.z);
  //printf("NHostAtom: %zu, HG_Nblock: %zu, NGuestAtom: %zu, GG_Nblock: %zu\n", NHostAtom, HG_Nblock, NGuestAtom, GG_Nblock);
  /*
  DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  /*
  DPCT1069:69: The argument 'FF' of the kernel function contains virtual
  pointer(s), which cannot be dereferenced. Try to migrate the code with
  "usm-level=restricted".
  */

  int3 NComp = SystemComponents.NComponents; 
//  q_ct1.submit([&](sycl::handler &cgh) {
//  sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1( sycl::range<1>(2 * HGGG_Nthread * sizeof(double)), cgh);
//
//  cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, HGGG_Nblock) *
//                                           sycl::range<3>(1, 1, HGGG_Nthread),
//                                       sycl::range<3>(1, 1, HGGG_Nthread)),
//                     [=](sycl::nd_item<3> item_ct1) {
//                       Calculate_Single_Body_Energy_SEPARATE_HostGuest_VDWReal(Sims.Box, Sims.d_a, Sims.Old, 
//				                                               Sims.New, FF, Sims.Blocksum, SelectedComponent, Atomsize, 
//						                                Molsize, Sims.device_flag, HG_Nblock, GG_Nblock, Do_New, Do_Old, NComp, item_ct1, dpct_local_acc_ct1.get_pointer());
//                     });
//  }).wait();


  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(sycl::range<1>(2 * HGGG_Nthread), cgh);

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, HGGG_Nblock) * sycl::range<3>(1, 1, HGGG_Nthread),
        sycl::range<3>(1, 1, HGGG_Nthread)),
        [=](sycl::nd_item<3> item_ct1) {
            Calculate_Single_Body_Energy_SEPARATE_HostGuest_VDWReal(
                Sims.Box, Sims.d_a, Sims.Old, Sims.New, FF, Sims.Blocksum,
                SelectedComponent, Atomsize, Molsize, Sims.device_flag,
                HG_Nblock, GG_Nblock, Do_New, Do_Old, NComp,
                item_ct1, dpct_local_acc_ct1.get_pointer());
        });
    }).wait();


  q_ct1.memcpy(Sims.flag, Sims.device_flag, sizeof(bool)).wait();

  MoveEnergy tot; bool Accept = false; double Pacc = 0.0;
  if(!Sims.flag[0])
  {
    double HGGG_BlockResult[HGGG_Nblock + HGGG_Nblock];
    q_ct1
        .memcpy(HGGG_BlockResult, Sims.Blocksum,
                2 * HGGG_Nblock * sizeof(double))
        .wait();
    double hg_vdw = 0.0; double gg_vdw = 0.0;
    double hg_real= 0.0; double gg_real= 0.0;
    for(size_t i = 0; i < HG_Nblock; i++)           hg_vdw += HGGG_BlockResult[i];
    for(size_t i = HG_Nblock; i < HGGG_Nblock; i++) gg_vdw += HGGG_BlockResult[i];

    for(size_t i = HGGG_Nblock; i < HG_Nblock + HGGG_Nblock; i++)
      hg_real += HGGG_BlockResult[i];
    for(size_t i = HGGG_Nblock + HG_Nblock; i < HGGG_Nblock + HGGG_Nblock; i++)
      gg_real += HGGG_BlockResult[i];
    /*
    printf("HG_NBlock: %zu\n", HGGG_Nblock);
    printf("Separated VDW : %.5f (HG), %.5f (GG)\n", hg_vdw , gg_vdw);
    printf("Separated Real: %.5f (HG), %.5f (GG)\n", hg_real, gg_real);
    */
    tot.HGVDW = hg_vdw; tot.HGReal = hg_real; tot.GGVDW = gg_vdw; tot.GGReal = gg_real;

    // Calculate Ewald //
    
    bool EwaldPerformed = false;
    if(!FF.noCharges && SystemComponents.hasPartialCharge[SelectedComponent])
    {
      double2 newScale = SystemComponents.Lambda[SelectedComponent].SET_SCALE(1.0);
      tot.EwaldE = GPU_EwaldDifference_General(Sims.Box, Sims.d_a, Sims.New, Sims.Old, FF, Sims.Blocksum, SystemComponents, SelectedComponent, MoveType, 0, newScale);
      tot.HGEwaldE=SystemComponents.tempdeltaHGEwald;
      EwaldPerformed = true;
    }
    double preFactor = GetPrefactor(SystemComponents, Sims, SelectedComponent, MoveType);
    Pacc = preFactor * std::exp(-SystemComponents.Beta * tot.total());

    //Apply the bias according to the macrostate//
    if(MoveType == SINGLE_INSERTION || MoveType == SINGLE_DELETION)
    {
      SystemComponents.Tmmc[SelectedComponent].ApplyWLBias(preFactor, SystemComponents.Beta, NMol, MoveType);
      SystemComponents.Tmmc[SelectedComponent].ApplyTMBias(preFactor, SystemComponents.Beta, NMol, MoveType);
    }
    //if(MoveType == SINGLE_INSERTION) printf("SINGLE INSERTION, tot: %.5f, preFactor: %.5f, Pacc: %.5f\n", tot.total(), preFactor, Pacc);
    //if(MoveType == SINGLE_DELETION)  printf("SINGLE DELETION,  tot: %.5f, preFactor: %.5f, Pacc: %.5f\n", tot.total(), preFactor, Pacc);
    if(Get_Uniform_Random() < preFactor * std::exp(-SystemComponents.Beta * tot.total())) Accept = true;
  }

  switch(MoveType)
  {
    case TRANSLATION:
    {
      if(Accept)
      {
        /*
        DPCT1049:2: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
      q_ct1.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, Molsize),
                                           sycl::range<3>(1, 1, Molsize)),
                         [=](sycl::nd_item<3> item_ct1) {
                           update_translation_position(
                               Sims.d_a, Sims.New, start_position,
                               SelectedComponent, item_ct1);
                         }).wait();
        SystemComponents.Moves[SelectedComponent].TranslationAccepted ++;
        if(!FF.noCharges && SystemComponents.hasPartialCharge[SelectedComponent])
        {
          Update_Ewald_Vector(Sims.Box, false, SystemComponents);
        }
      }
      else {tot.zero(); };
      SystemComponents.Tmmc[SelectedComponent].Update(1.0, NMol, MoveType);
      break;
    }
    case ROTATION:
    {
      if(Accept)
      {
        /*
        DPCT1049:3: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
      q_ct1.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, Molsize),
                                           sycl::range<3>(1, 1, Molsize)),
                         [=](sycl::nd_item<3> item_ct1) {
                           update_translation_position(
                               Sims.d_a, Sims.New, start_position,
                               SelectedComponent, item_ct1);
                         }).wait();
        SystemComponents.Moves[SelectedComponent].RotationAccepted ++;
        if(!FF.noCharges && SystemComponents.hasPartialCharge[SelectedComponent])
        {
          Update_Ewald_Vector(Sims.Box, false, SystemComponents);
        }
      }
      else {tot.zero(); };
      SystemComponents.Tmmc[SelectedComponent].Update(1.0, NMol, MoveType);
      break;
    }
    case SINGLE_INSERTION:
    {
      SystemComponents.Tmmc[SelectedComponent].TreatAccOutofBound(Accept, NMol, MoveType);
      if(Accept)
      {
        SystemComponents.Moves[SelectedComponent].InsertionAccepted ++;
        AcceptInsertion(SystemComponents, Sims, SelectedComponent, 0, FF.noCharges, SINGLE_INSERTION); //0: selectedTrial//
      }
      else {tot.zero(); };
      SystemComponents.Tmmc[SelectedComponent].Update(Pacc, NMol, MoveType);
      break;
    }
    case SINGLE_DELETION:
    {
      SystemComponents.Tmmc[SelectedComponent].TreatAccOutofBound(Accept, NMol, MoveType);
      if(Accept)
      {
        SystemComponents.Moves[SelectedComponent].DeletionAccepted ++;
        size_t UpdateLocation = SelectedMolInComponent * SystemComponents.Moleculesize[SelectedComponent];
        AcceptDeletion(SystemComponents, Sims, SelectedComponent, UpdateLocation, SelectedMolInComponent, FF.noCharges);
      }
      else {tot.zero(); };
      SystemComponents.Tmmc[SelectedComponent].Update(Pacc, NMol, MoveType);
      break;
    }
  }
  return tot;
}
