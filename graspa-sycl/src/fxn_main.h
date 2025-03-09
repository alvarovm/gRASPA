#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cmath>
#include <algorithm>
void Setup_RandomNumber(RandomNumber &Random, size_t SIZE);
void Copy_Atom_data_to_device(size_t NumberOfComponents, Atoms* device_System, Atoms* System);
void Update_Components_for_framework(size_t NumberOfComponents, Components& SystemComponents, Atoms* System);

void Setup_Temporary_Atoms_Structure(Atoms& TempMol, Atoms* System);  

void Initialize_Move_Statistics(Move_Statistics& MoveStats);

void Setup_Box_Temperature_Pressure(Units& Constants, Components& SystemComponents, Boxsize& device_Box);

void Prepare_ForceField(ForceField& FF, ForceField& device_FF, PseudoAtomDefinitions PseudoAtom);

void Prepare_Widom(WidomStruct& Widom, Boxsize Box, Simulations& Sims, Components SystemComponents, Atoms* System, Move_Statistics MoveStats);

template<typename T>
T* CUDA_allocate_array(size_t N, T InitVal)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  T* device_x;
  device_x = (T *)sycl::malloc_device(N * sizeof(T), q_ct1);
      checkCUDAError("Error allocating Malloc");
  T array[N];
  for(size_t i = 0; i < N; i++) array[i] = InitVal;
  q_ct1.memcpy(device_x, array, N * sizeof(T)).wait();
  //cudaMemset(device_x, (T) InitVal, N * sizeof(T));
  return device_x;
}

template<typename T>
T* CUDA_copy_allocate_array(T* x, size_t N)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  T* device_x;
  device_x = (T *)sycl::malloc_device(N * sizeof(T), q_ct1);
  q_ct1.memcpy(device_x, x, N * sizeof(T)).wait();
  return device_x;
}

void Prepare_TempSystem_On_Host(Atoms& TempSystem)
{
    size_t Allocate_size = 1024;
    /*
    DPCT1083:45: The size of double3 in the migrated code may be different from
    the original code. Check that the allocated memory size in the migrated code
    is correct.
    */
    TempSystem.pos =
        (double3 *)malloc(Allocate_size * sizeof(double3));
    TempSystem.scale         = (double*) malloc(Allocate_size*sizeof(double));
    TempSystem.charge        = (double*) malloc(Allocate_size*sizeof(double));
    TempSystem.scaleCoul     = (double*) malloc(Allocate_size*sizeof(double));
    TempSystem.Type          = (size_t*) malloc(Allocate_size*sizeof(size_t));
    TempSystem.MolID         = (size_t*) malloc(Allocate_size*sizeof(size_t));
    TempSystem.size          = 0;
    TempSystem.Allocate_size = Allocate_size;
}

inline void Setup_RandomNumber(RandomNumber& Random, size_t SIZE)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  Random.randomsize = SIZE; Random.offset = 0;
  /*
  DPCT1083:46: The size of double3 in the migrated code may be different from
  the original code. Check that the allocated memory size in the migrated code
  is correct.
  */
  Random.host_random =
      (double3 *)malloc(Random.randomsize * sizeof(double3));
  for (size_t i = 0; i < Random.randomsize; i++)
  {
    Random.host_random[i].x() = Get_Uniform_Random();
    Random.host_random[i].y() = Get_Uniform_Random();
    Random.host_random[i].z() = Get_Uniform_Random();
  }
  //Add some padding to match the sequence of the previous code, pad it up to 1 million numbers//
  for (size_t i = Random.randomsize * 3; i < 1000000; i++) Get_Uniform_Random();

  /*
  DPCT1083:47: The size of double3 in the migrated code may be different from
  the original code. Check that the allocated memory size in the migrated code
  is correct.
  */
  (Random.device_random) =
      (typename std::remove_reference<decltype(Random.device_random)>::type)
          sycl::malloc_device(Random.randomsize * sizeof(double3), q_ct1);
  /*
  DPCT1083:48: The size of double3 in the migrated code may be different from
  the original code. Check that the allocated memory size in the migrated code
  is correct.
  */
  q_ct1
      .memcpy(Random.device_random, Random.host_random,
              Random.randomsize * sizeof(double3))
      .wait();
}

inline void Copy_Atom_data_to_device(size_t NumberOfComponents, Atoms* device_System, Atoms* System)
{
  size_t required_size = 0;
  for(size_t i = 0; i < NumberOfComponents; i++)
  {
    if(i == 0){ required_size = System[i].size;} else { required_size = System[i].Allocate_size; }
    device_System[i].pos           = CUDA_copy_allocate_array(System[i].pos,       required_size);
    device_System[i].scale         = CUDA_copy_allocate_array(System[i].scale,     required_size);
    device_System[i].charge        = CUDA_copy_allocate_array(System[i].charge,    required_size);
    device_System[i].scaleCoul     = CUDA_copy_allocate_array(System[i].scaleCoul, required_size);
    device_System[i].Type          = CUDA_copy_allocate_array(System[i].Type,      required_size);
    device_System[i].MolID         = CUDA_copy_allocate_array(System[i].MolID,     required_size);
    device_System[i].size          = System[i].size;
    device_System[i].Molsize       = System[i].Molsize;
    device_System[i].Allocate_size = System[i].Allocate_size;
  }
}

inline void Update_Components_for_framework(size_t NumberOfComponents, Components& SystemComponents, Atoms* System)
{
  SystemComponents.Total_Components = NumberOfComponents; //Framework + 1 adsorbate
  SystemComponents.TotalNumberOfMolecules = 1; //If there is a framework, framework is counted as a molecule//
  SystemComponents.NumberOfFrameworks = 1; //Just one framework
  SystemComponents.MoleculeName.push_back("MOF"); //Name of the framework
  SystemComponents.Moleculesize.push_back(System[0].size);
  SystemComponents.Allocate_size.push_back(System[0].size);
  SystemComponents.NumberOfMolecule_for_Component.push_back(1);
  SystemComponents.MolFraction.push_back(1.0);
  SystemComponents.IdealRosenbluthWeight.push_back(1.0);
  SystemComponents.FugacityCoeff.push_back(1.0);
  SystemComponents.Tc.push_back(0.0);        //Tc for framework is set to zero
  SystemComponents.Pc.push_back(0.0);        //Pc for framework is set to zero
  SystemComponents.Accentric.push_back(0.0); //Accentric factor for framework is set to zero
  //Zhao's note: for now, assume the framework is rigid//
  SystemComponents.rigid.push_back(true);
  SystemComponents.hasfractionalMolecule.push_back(false); //No fractional molecule for the framework//
  SystemComponents.NumberOfCreateMolecules.push_back(0); //Create zero molecules for the framework//
  LAMBDA lambda;
  lambda.newBin = 0; lambda.delta = static_cast<double>(1.0/(lambda.binsize)); lambda.WangLandauScalingFactor = 0.0; //Zhao's note: in raspa3, delta is 1/(nbin - 1)
  lambda.FractionalMoleculeID = 0;
  SystemComponents.Lambda.push_back(lambda);

  TMMC tmmc;
  SystemComponents.Tmmc.push_back(tmmc); //Just use default values for tmmc for the framework, it will do nothing//
  //Add PseudoAtoms from the Framework to the total PseudoAtoms array//
  SystemComponents.UpdatePseudoAtoms(INSERTION, 0);
}

inline void Setup_Temporary_Atoms_Structure(Atoms& TempMol, Atoms* System)
{
  //Set up MolArrays//
  size_t Allocate_size_Temporary=1024; //Assign 1024 empty slots for the temporary structures//
  //OLD//
  TempMol.pos = CUDA_allocate_array<double3>(Allocate_size_Temporary,
                                                   {0.0, 0.0, 0.0});
  TempMol.scale     = CUDA_allocate_array<double>  (Allocate_size_Temporary, 0.0);
  TempMol.charge    = CUDA_allocate_array<double>  (Allocate_size_Temporary, 0.0);
  TempMol.scaleCoul = CUDA_allocate_array<double>  (Allocate_size_Temporary, 0.0);
  TempMol.Type      = CUDA_allocate_array<size_t>  (Allocate_size_Temporary, 0.0);
  TempMol.MolID     = CUDA_allocate_array<size_t>  (Allocate_size_Temporary, 0.0);
  TempMol.size      = 0;
  TempMol.Molsize   = 0;
  TempMol.Allocate_size = Allocate_size_Temporary;
}

inline void Initialize_Move_Statistics(Move_Statistics& MoveStats)
{
  MoveStats.TranslationProb = 0.0; MoveStats.RotationProb = 0.0; MoveStats.WidomProb = 0.0; MoveStats.SwapProb = 0.0; MoveStats.ReinsertionProb = 0.0; MoveStats.CBCFProb = 0.0;
  MoveStats.TranslationAccepted = 0; MoveStats.TranslationTotal      = 0; MoveStats.TranslationAccRatio = 0.0;
  MoveStats.RotationAccepted    = 0; MoveStats.RotationTotal         = 0; MoveStats.RotationAccRatio    = 0.0;
  MoveStats.InsertionAccepted   = 0; MoveStats.InsertionTotal        = 0; 
  MoveStats.DeletionAccepted    = 0; MoveStats.DeletionTotal         = 0;
  MoveStats.ReinsertionAccepted = 0; MoveStats.ReinsertionTotal      = 0;
  MoveStats.CBCFAccepted        = 0; MoveStats.CBCFTotal             = 0; 
  MoveStats.CBCFInsertionTotal  = 0; MoveStats.CBCFInsertionAccepted = 0;
  MoveStats.CBCFLambdaTotal     = 0; MoveStats.CBCFLambdaAccepted    = 0;
  MoveStats.CBCFDeletionTotal   = 0; MoveStats.CBCFDeletionAccepted  = 0;
}

inline void Setup_Box_Temperature_Pressure(Units& Constants, Components& SystemComponents, Boxsize& device_Box)
{
  SystemComponents.Beta =
      1.0 / (Constants.BoltzmannConstant /
             (Constants.MassUnit * pow(Constants.LengthUnit, 2) /
              pow(Constants.TimeUnit, 2)) *
             SystemComponents.Temperature);
  //Convert pressure from pascal
  device_Box.Pressure /= (Constants.MassUnit /
                          (Constants.LengthUnit * pow(Constants.TimeUnit, 2)));
  printf("------------------- SIMULATION BOX PARAMETERS -----------------\n");
  printf("Pressure:        %.5f\n", device_Box.Pressure);
  printf("Box Volume:      %.5f\n", device_Box.Volume);
  printf("Box Beta:        %.5f\n", SystemComponents.Beta);
  printf("Box Temperature: %.5f\n", SystemComponents.Temperature);
  printf("---------------------------------------------------------------\n");
}

inline void Prepare_ForceField(ForceField& FF, ForceField& device_FF, PseudoAtomDefinitions PseudoAtom)
{
  // COPY DATA TO DEVICE POINTER //
  //device_FF.FFParams      = CUDA_copy_allocate_array(FF.FFParams, 5);
  device_FF.OverlapCriteria = FF.OverlapCriteria;
  device_FF.CutOffVDW       = FF.CutOffVDW;
  device_FF.CutOffCoul      = FF.CutOffCoul;
  //device_FF.Prefactor       = FF.Prefactor;
  //device_FF.Alpha           = FF.Alpha;

  device_FF.epsilon         = CUDA_copy_allocate_array(FF.epsilon, FF.size*FF.size);
  device_FF.sigma           = CUDA_copy_allocate_array(FF.sigma, FF.size*FF.size);
  device_FF.z               = CUDA_copy_allocate_array(FF.z, FF.size*FF.size);
  device_FF.shift           = CUDA_copy_allocate_array(FF.shift, FF.size*FF.size);
  device_FF.FFType          = CUDA_copy_allocate_array(FF.FFType, FF.size*FF.size);
  device_FF.noCharges       = FF.noCharges;
  device_FF.size            = FF.size;
  device_FF.VDWRealBias     = FF.VDWRealBias;
  //Formulate Component statistics on the host
  //ForceFieldParser(FF, PseudoAtom);
  //PseudoAtomParser(FF, PseudoAtom);
}

inline void Prepare_Widom(WidomStruct& Widom, Boxsize Box, Simulations& Sims, Components SystemComponents, Atoms* System)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  //Zhao's note: NumberWidomTrials is for first bead. NumberWidomTrialsOrientations is for the rest, here we consider single component, not mixture //

  size_t MaxTrialsize = std::max(Widom.NumberWidomTrials, Widom.NumberWidomTrialsOrientations*(SystemComponents.Moleculesize[1]-1));

  //Zhao's note: The previous way yields a size for blocksum that can be smaller than the number of kpoints
  //This is a problem when you need to do parallel Ewald summation for the whole system//
  //Might be good to add a flag or so//
  //size_t MaxResultsize = MaxTrialsize*(System[0].Allocate_size+System[1].Allocate_size);
  size_t MaxAllocatesize =
      std::max(System[0].Allocate_size, System[1].Allocate_size);
  size_t MaxResultsize = MaxTrialsize * SystemComponents.Total_Components * MaxAllocatesize * 5; //For Volume move, it really needs a lot of blocks//


  printf("----------------- MEMORY ALLOCAION STATUS -----------------\n");
  //Compare Allocate sizes//
  printf("System allocate_sizes are: %zu, %zu\n", System[0].Allocate_size, System[1].Allocate_size); 
  printf("Component allocate_sizes are: %zu, %zu\n", SystemComponents.Allocate_size[0], SystemComponents.Allocate_size[1]);

  Sims.flag        = (bool*)malloc(MaxTrialsize * sizeof(bool));
  (Sims.device_flag) =
      (typename std::remove_reference<decltype(Sims.device_flag)>::type)
          sycl::malloc_host(MaxTrialsize * sizeof(bool), q_ct1);

  size_t NNN = (MaxResultsize/DEFAULTTHREAD + 1);
  (Sims.Blocksum) =
      (typename std::remove_reference<decltype(Sims.Blocksum)>::type)
          sycl::malloc_host(NNN * sizeof(double), q_ct1);

  Sims.temp = sycl::malloc_device<double3>(100, q_ct1);
  //Sims.Blocksum = sycl::malloc_device<double>(NNN, q_ct1);
  //cudaMallocHost(&Sims.Blocksum,             (MaxResultsize/DEFAULTTHREAD + 1)*sizeof(double));

  (Sims.ExcludeList) =
      (typename std::remove_reference<decltype(Sims.ExcludeList)>::type)
          sycl::malloc_shared(10 * sizeof(int2), q_ct1);
  for(size_t i = 0; i < 10; i++) Sims.ExcludeList[i] = {-1, -1}; //Initialize with negative # so that nothing is ignored//
  //cudaMalloc(&Sims.Blocksum,             (MaxResultsize/DEFAULTTHREAD + 1)*sizeof(double));

  printf("Allocated Blocksum size: %zu\n", (MaxResultsize/DEFAULTTHREAD + 1));
 
  //cudaMalloc(&Sims.Blocksum,             (MaxResultsize/DEFAULTTHREAD + 1)*sizeof(double));
  Sims.Nblocks = MaxResultsize/DEFAULTTHREAD + 1;

  printf("Allocated %zu doubles for Blocksums\n", MaxResultsize/DEFAULTTHREAD + 1);

  std::vector<double> MaxRotation    = {30.0/(180/3.1415), 30.0/(180/3.1415), 30.0/(180/3.1415)};
  Sims.MaxTranslation.x() = Box.Cell[0] * 0.1;
      Sims.MaxTranslation.y() = Box.Cell[4] * 0.1;
      Sims.MaxTranslation.z() = Box.Cell[8] * 0.1;
  Sims.MaxRotation.x() = 30.0 / (180 / 3.1415);
      Sims.MaxRotation.y() = 30.0 / (180 / 3.1415);
      Sims.MaxRotation.z() = 30.0 / (180 / 3.1415);

  Sims.start_position = 0;
  //Sims.Nblocks = 0;
  Sims.TotalAtoms = 0;
  Sims.AcceptedFlag = false;

  Widom.WidomFirstBeadAllocatesize = MaxResultsize/DEFAULTTHREAD;
  printf("------------------------------------------------------------\n");
}

inline void Allocate_Copy_Ewald_Vector(Boxsize& device_Box, Components SystemComponents)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  printf("******   Allocating Ewald WaveVectors (INITIAL STAGE ONLY)   ******\n");
  //Zhao's note: This only works if the box size is not changed, eik_xy might not be useful if box size is not changed//
  size_t eikx_size     = SystemComponents.eik_x.size() * 2;
  size_t eiky_size     = SystemComponents.eik_y.size() * 2; //added times 2 for box volume move//
  size_t eikz_size     = SystemComponents.eik_z.size() * 2;
  printf("Allocated %zu %zu %zu space for eikxyz\n", eikx_size, eiky_size, eikz_size);
  //size_t eikxy_size    = SystemComponents.eik_xy.size();
  size_t storedEiksize = SystemComponents.storedEik.size() * 2; //added times 2 for box volume move//
  device_Box.eik_x        = sycl::malloc_device<Complex>(eikx_size, q_ct1);
  device_Box.eik_y        = sycl::malloc_device<Complex>(eiky_size, q_ct1);
  device_Box.eik_z        = sycl::malloc_device<Complex>(eikz_size, q_ct1);
  device_Box.storedEik    = sycl::malloc_device<Complex>(storedEiksize, q_ct1);
  device_Box.totalEik     = sycl::malloc_device<Complex>(storedEiksize, q_ct1);
  device_Box.FrameworkEik = sycl::malloc_device<Complex>(storedEiksize, q_ct1);

  Complex storedEik[storedEiksize]; //Temporary Complex struct on the host//
  Complex FrameworkEik[storedEiksize];
  //for(size_t i = 0; i < SystemComponents.storedEik.size(); i++)
  for(size_t i = 0; i < storedEiksize; i++)
  {
    if(i < SystemComponents.storedEik.size())
    {
      storedEik[i].real = SystemComponents.storedEik[i].real();
      storedEik[i].imag = SystemComponents.storedEik[i].imag();
      
      FrameworkEik[i].real = SystemComponents.FrameworkEik[i].real();
      FrameworkEik[i].imag = SystemComponents.FrameworkEik[i].imag();
    }
    else
    {
      storedEik[i].real    = 0.0; storedEik[i].imag    = 0.0;
      FrameworkEik[i].real = 0.0; FrameworkEik[i].imag = 0.0;
    }
    if(i < 10) printf("Wave Vector %zu is %.5f %.5f\n", i, storedEik[i].real, storedEik[i].imag);
  }
  q_ct1.memcpy(device_Box.storedEik, storedEik, storedEiksize * sizeof(Complex))
      .wait();
  q_ct1
      .memcpy(device_Box.FrameworkEik, FrameworkEik,
              storedEiksize * sizeof(Complex))
      .wait();
  printf("****** DONE Allocating Ewald WaveVectors (INITIAL STAGE ONLY) ******\n");
}

inline void Check_Simulation_Energy(Boxsize& Box, Atoms* System, ForceField FF, ForceField device_FF, Components& SystemComponents, int SIMULATIONSTAGE, size_t Numsim, Simulations& Sim)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::string STAGE; 
  switch(SIMULATIONSTAGE)
  {
    case INITIAL:
    { STAGE = "INITIAL"; break;}
    case CREATEMOL:
    { STAGE = "CREATE_MOLECULE"; break;}
    case FINAL:
    { STAGE = "FINAL"; break;}
  }
  printf("======================== CALCULATING %s STAGE ENERGY ========================\n", STAGE.c_str());
  MoveEnergy ENERGY;
  Atoms device_System[SystemComponents.Total_Components];
  q_ct1
      .memcpy(device_System, Sim.d_a,
              SystemComponents.Total_Components * sizeof(Atoms))
      .wait();
  q_ct1.memcpy(Box.Cell, Sim.Box.Cell, 9 * sizeof(double)).wait();
  q_ct1.memcpy(Box.InverseCell, Sim.Box.InverseCell, 9 * sizeof(double)).wait();
  //Update every value that can be changed during a volume move//
  Box.Volume = Sim.Box.Volume;
  Box.ReciprocalCutOff = Sim.Box.ReciprocalCutOff;
  Box.Cubic = Sim.Box.Cubic;
  Box.kmax  = Sim.Box.kmax;

  MoveEnergy GPU_Energy;

  double start = omp_get_wtime();
  VDWReal_Total_CPU(Box, System, device_System, FF, SystemComponents, ENERGY);
  ENERGY.print();
  double end = omp_get_wtime();
  double CPUSerialTime = end - start;
         start = omp_get_wtime();
  double* xxx; xxx = (double*) malloc(sizeof(double)*2);
  double* device_xxx; device_xxx = CUDA_copy_allocate_array(xxx, 2);
  //Zhao's note: if the serial GPU energy test is too slow, comment it out//
  //one_thread_GPU_test<<<1,1>>>(Sim.Box, Sim.d_a, device_FF, device_xxx);
  q_ct1.memcpy(xxx, device_xxx, sizeof(double)).wait();
         end = omp_get_wtime();
  dev_ct1.queues_wait_and_throw();

  double SerialGPUTime = end - start;
  //For total energy, divide the parallelization into several parts//
  //For framework, every thread treats the interaction between one framework atom with an adsorbate molecule//
  //For adsorbate/adsorbate, every thread treats one adsorbate molecule with an adsorbate molecule//
  start = omp_get_wtime();
  size_t Host_threads  = 0;
  size_t Guest_threads = 0;
  size_t NFrameworkAtomsPerThread = 4;
  size_t NAdsorbate = 0;
  for(size_t i = 1; i < SystemComponents.Total_Components; i++) NAdsorbate += SystemComponents.NumberOfMolecule_for_Component[i];
  Host_threads  = SystemComponents.Moleculesize[0] / NFrameworkAtomsPerThread; //Per adsorbate molecule//
  if(SystemComponents.Moleculesize[0] % NFrameworkAtomsPerThread != 0) Host_threads ++;
  Host_threads *= NAdsorbate; //Total = Host_thread_per_molecule * number of Adsorbate molecule
  Guest_threads = NAdsorbate * (NAdsorbate-1)/2;
  if(Host_threads + Guest_threads > 0)
  {
    bool   ConsiderHostHost = false;
    bool   UseOffset        = false;
    GPU_Energy += Total_VDW_Coulomb_Energy(Sim, device_FF, NAdsorbate, Host_threads, Guest_threads, NFrameworkAtomsPerThread, ConsiderHostHost, UseOffset);
  }
  end = omp_get_wtime();

  //Do Parallel Total Ewald//
  double TotEwald     = 0.0;
  double CPUEwaldTime = 0.0;
  double GPUEwaldTime = 0.0;

  if(!device_FF.noCharges)
  {
    dev_ct1.queues_wait_and_throw();
    double EwStart = omp_get_wtime();

    CPU_GPU_EwaldTotalEnergy(Box, Sim.Box, System, Sim.d_a, FF, device_FF, SystemComponents, ENERGY);
    ENERGY.EwaldE -= SystemComponents.FrameworkEwald;

    double EwEnd  = omp_get_wtime();
    //Zhao's note: if it is in the initial stage, calculate the intra and self exclusion energy for ewald summation//
    if(SIMULATIONSTAGE == INITIAL) Calculate_Exclusion_Energy_Rigid(Box, System, FF, SystemComponents);
    CPUEwaldTime = EwEnd - EwStart;

    dev_ct1.queues_wait_and_throw();
    //Zhao's note: if doing initial energy, initialize and copy host Ewald to device// 
    if(SIMULATIONSTAGE == INITIAL) Allocate_Copy_Ewald_Vector(Sim.Box, SystemComponents);
    Check_WaveVector_CPUGPU(Sim.Box, SystemComponents); //Check WaveVector on the CPU and GPU//
    dev_ct1.queues_wait_and_throw();
    EwStart = omp_get_wtime();
    bool UseOffset = false;
    //GPU_Energy  += Ewald_TotalEnergy(Sim, SystemComponents, UseOffset);
    GPU_Energy.EwaldE  -= SystemComponents.FrameworkEwald;
    dev_ct1.queues_wait_and_throw();
    EwEnd = omp_get_wtime();
    GPUEwaldTime = EwEnd - EwStart;
  }

  //Calculate Tail Correction Energy//
  ENERGY.TailE = TotalTailCorrection(SystemComponents, FF.size, Sim.Box.Volume);

  //ENERGY.DNN_E = Predict_From_FeatureMatrix_Total(Sim, SystemComponents);
 
  if(SystemComponents.UseDNNforHostGuest) double Correction = ENERGY.DNN_Correction();
 
  if(SIMULATIONSTAGE == INITIAL) SystemComponents.Initial_Energy = ENERGY;
  else if(SIMULATIONSTAGE == CREATEMOL)
  {
    SystemComponents.CreateMol_Energy = ENERGY;
  }
  else
  { 
    SystemComponents.Final_Energy = ENERGY;
  }
  printf("====================== DONE CALCULATING %s STAGE ENERGY ======================\n", STAGE.c_str());
}

inline void Copy_AtomData_from_Device(Atoms* System, Atoms* Host_System, Atoms* d_a, Components& SystemComponents)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  q_ct1.memcpy(System, d_a, SystemComponents.Total_Components * sizeof(Atoms))
      .wait();
  //printval<<<1,1>>>(System[0]);
  //printvald_a<<<1,1>>>(d_a);
  for(size_t ijk=0; ijk < SystemComponents.Total_Components; ijk++)
  {
    // if the host allocate_size is different from the device, allocate more space on the host
    /*
    DPCT1083:49: The size of double3 in the migrated code may be different from
    the original code. Check that the allocated memory size in the migrated code
    is correct.
    */
    Host_System[ijk].pos = (double3 *)malloc(System[ijk].Allocate_size *
                                                   sizeof(double3));
    Host_System[ijk].scale     = (double*)  malloc(System[ijk].Allocate_size*sizeof(double));
    Host_System[ijk].charge    = (double*)  malloc(System[ijk].Allocate_size*sizeof(double));
    Host_System[ijk].scaleCoul = (double*)  malloc(System[ijk].Allocate_size*sizeof(double));
    Host_System[ijk].Type      = (size_t*)  malloc(System[ijk].Allocate_size*sizeof(size_t));
    Host_System[ijk].MolID     = (size_t*)  malloc(System[ijk].Allocate_size*sizeof(size_t));
    Host_System[ijk].size      = System[ijk].size;
    Host_System[ijk].Allocate_size = System[ijk].Allocate_size;

    /*
    DPCT1083:50: The size of double3 in the migrated code may be different from
    the original code. Check that the allocated memory size in the migrated code
    is correct.
    */
    q_ct1
        .memcpy(Host_System[ijk].pos, System[ijk].pos,
                sizeof(double3) * System[ijk].Allocate_size)
        .wait();
    q_ct1
        .memcpy(Host_System[ijk].scale, System[ijk].scale,
                sizeof(double) * System[ijk].Allocate_size)
        .wait();
    q_ct1
        .memcpy(Host_System[ijk].charge, System[ijk].charge,
                sizeof(double) * System[ijk].Allocate_size)
        .wait();
    q_ct1
        .memcpy(Host_System[ijk].scaleCoul, System[ijk].scaleCoul,
                sizeof(double) * System[ijk].Allocate_size)
        .wait();
    q_ct1
        .memcpy(Host_System[ijk].Type, System[ijk].Type,
                sizeof(size_t) * System[ijk].Allocate_size)
        .wait();
    q_ct1
        .memcpy(Host_System[ijk].MolID, System[ijk].MolID,
                sizeof(size_t) * System[ijk].Allocate_size)
        .wait();
    Host_System[ijk].size = System[ijk].size;
  }
}

inline void PRINT_ENERGY_AT_STAGE(Components& SystemComponents, int stage, Units& Constants)
{
  std::string stage_name;
  MoveEnergy  E;
  switch(stage)
  {
    case INITIAL:               {stage_name = "INITIAL STAGE";         E = SystemComponents.Initial_Energy;   break;}
    case CREATEMOL:             {stage_name = "CREATE MOLECULE STAGE"; E = SystemComponents.CreateMol_Energy; break;}
    case FINAL:                 {stage_name = "FINAL STAGE";           E = SystemComponents.Final_Energy;     break;}
    case CREATEMOL_DELTA:       {stage_name = "RUNNING DELTA_E (CREATE MOLECULE - INITIAL)"; E = SystemComponents.CreateMoldeltaE; break;}
    case DELTA:                 {stage_name = "RUNNING DELTA_E (FINAL - CREATE MOLECULE)";   E = SystemComponents.deltaE; break;}
    case CREATEMOL_DELTA_CHECK: {stage_name = "CHECK DELTA_E (CREATE MOLECULE - INITIAL)"; E = SystemComponents.CreateMol_Energy - SystemComponents.Initial_Energy; break;}
    case DELTA_CHECK: {stage_name = "CHECK DELTA_E (FINAL - CREATE MOLECULE)"; E = SystemComponents.Final_Energy - SystemComponents.CreateMol_Energy; break;}
    case DRIFT: {stage_name = "ENERGY DRIFT"; E = SystemComponents.CreateMol_Energy + SystemComponents.deltaE - SystemComponents.Final_Energy; break;}
  }
  printf(" *** %s *** \n", stage_name.c_str());
  printf("========================================================================\n");
  printf("VDW [Host-Guest]:           %.5f (%.5f [K])\n", E.HGVDW, E.HGVDW * Constants.energy_to_kelvin); 
  printf("VDW [Guest-Guest]:          %.5f (%.5f [K])\n", E.GGVDW, E.GGVDW * Constants.energy_to_kelvin);
  printf("Real Coulomb [Host-Guest]:  %.5f (%.5f [K])\n", E.HGReal, E.HGReal * Constants.energy_to_kelvin); 
  printf("Real Coulomb [Guest-Guest]: %.5f (%.5f [K])\n", E.GGReal, E.GGReal * Constants.energy_to_kelvin);
  printf("Ewald [Host-Guest]:         %.5f (%.5f [K])\n", E.HGEwaldE, E.HGEwaldE * Constants.energy_to_kelvin);
  printf("Ewald [Guest-Guest]:        %.5f (%.5f [K])\n", E.EwaldE, E.EwaldE * Constants.energy_to_kelvin);
  printf("DNN Energy:                 %.5f (%.5f [K])\n", E.DNN_E, E.DNN_E * Constants.energy_to_kelvin);
  if(SystemComponents.UseDNNforHostGuest)
  {
    printf(" --> Stored Classical Host-Guest Interactions: \n");
    printf("     VDW:             %.5f (%.5f [K])\n", E.storedHGVDW, E.storedHGVDW * Constants.energy_to_kelvin);
    printf("     Real Coulomb:    %.5f (%.5f [K])\n", E.storedHGReal, E.storedHGReal * Constants.energy_to_kelvin);
    printf("     Ewald:           %.5f (%.5f [K])\n", E.storedHGEwaldE, E.storedHGEwaldE * Constants.energy_to_kelvin);
    printf("     Total:           %.5f (%.5f [K])\n", E.storedHGVDW + E.storedHGReal + E.storedHGEwaldE, (E.storedHGVDW + E.storedHGReal + E.storedHGEwaldE) * Constants.energy_to_kelvin);
    printf(" --> DNN - Classical: %.5f (%.5f [K])\n", E.DNN_E - (E.storedHGVDW + E.storedHGReal + E.storedHGEwaldE), (E.DNN_E - (E.storedHGVDW + E.storedHGReal + E.storedHGEwaldE)) * Constants.energy_to_kelvin);
  }
  printf("Tail Correction Energy:     %.5f (%.5f [K])\n", E.TailE, E.TailE * Constants.energy_to_kelvin);
  printf("Total Energy:               %.5f (%.5f [K])\n", E.total(), E.total() * Constants.energy_to_kelvin);
  printf("========================================================================\n");
}
inline void ENERGY_SUMMARY(std::vector<Components>& SystemComponents, Units& Constants)
{
  size_t NumberOfSimulations = SystemComponents.size();
  for(size_t i = 0; i < NumberOfSimulations; i++)
  {
    printf("======================== ENERGY SUMMARY (Simulation %zu) =========================\n", i);
    PRINT_ENERGY_AT_STAGE(SystemComponents[i], INITIAL, Constants);
    PRINT_ENERGY_AT_STAGE(SystemComponents[i], CREATEMOL, Constants);
    PRINT_ENERGY_AT_STAGE(SystemComponents[i], CREATEMOL_DELTA, Constants);
    PRINT_ENERGY_AT_STAGE(SystemComponents[i], CREATEMOL_DELTA_CHECK, Constants);
    PRINT_ENERGY_AT_STAGE(SystemComponents[i], FINAL, Constants);
    PRINT_ENERGY_AT_STAGE(SystemComponents[i], DELTA, Constants);
    PRINT_ENERGY_AT_STAGE(SystemComponents[i], DELTA_CHECK, Constants);
    PRINT_ENERGY_AT_STAGE(SystemComponents[i], DRIFT, Constants);
    printf("================================================================================\n");

    printf("DNN Rejection Summary:\nTranslation+Rotation: %zu\nReinsertion: %zu\nInsertion: %zu\nDeletion: %zu\nSingleSwap: %zu\n", SystemComponents[i].TranslationRotationDNNReject, SystemComponents[i].ReinsertionDNNReject, SystemComponents[i].InsertionDNNReject, SystemComponents[i].DeletionDNNReject, SystemComponents[i].SingleSwapDNNReject);
    printf("DNN Drift Summary:\nTranslation+Rotation: %.5f\nReinsertion: %.5f\nInsertion: %.5f\nDeletion: %.5f\nSingleSwap: %.5f\n", SystemComponents[i].SingleMoveDNNDrift, SystemComponents[i].ReinsertionDNNDrift, SystemComponents[i].InsertionDNNDrift, SystemComponents[i].DeletionDNNDrift, SystemComponents[i].SingleSwapDNNDrift);
  }
}

inline void GenerateRestartMovies(int Cycle, std::vector<Components>& SystemComponents, std::vector<Simulations>& Sims, ForceField& FF, std::vector<Boxsize>& Box, PseudoAtomDefinitions& PseudoAtom)
{
  
  size_t NumberOfSimulations = SystemComponents.size();
  for(size_t i = 0; i < NumberOfSimulations; i++)
  {
    printf("System %zu\n", i);
    Atoms device_System[SystemComponents[i].Total_Components];
    Copy_AtomData_from_Device(device_System, SystemComponents[i].HostSystem, Sims[i].d_a, SystemComponents[i]);

    create_movie_file(Cycle, SystemComponents[i].HostSystem, SystemComponents[i], FF, Box[i], PseudoAtom.Name, i);
    create_Restart_file(Cycle, SystemComponents[i].HostSystem, SystemComponents[i], FF, Box[i], PseudoAtom.Name, Sims[i].MaxTranslation, Sims[i].MaxRotation, i);
    Write_All_Adsorbate_data(Cycle, SystemComponents[i].HostSystem, SystemComponents[i], FF, Box[i], PseudoAtom.Name, i);
    Write_Lambda(Cycle, SystemComponents[i], i);
    Write_TMMC(Cycle, SystemComponents[i], i);
    //Print Number of Pseudo Atoms//
    for(size_t j = 0; j < SystemComponents[i].NumberOfPseudoAtoms.size(); j++) printf("PseudoAtom Type: %s[%zu], #: %zu\n", PseudoAtom.Name[j].c_str(), j, SystemComponents[i].NumberOfPseudoAtoms[j]);
  }
}

inline void prepare_MixtureStats(Components& SystemComponents)
{
  double tot = 0.0;
  printf("================= MOL FRACTIONS =================\n");
  for(size_t j = 0; j < SystemComponents.Total_Components; j++)
  {
    SystemComponents.Moves[j].IdentitySwap_Total_TO.resize(SystemComponents.Total_Components, 0);
    SystemComponents.Moves[j].IdentitySwap_Acc_TO.resize(SystemComponents.Total_Components,   0);
    if(j != 0) tot += SystemComponents.MolFraction[j];
  }
  //Prepare MolFraction for adsorbate components//
  for(size_t j = 1; j < SystemComponents.Total_Components; j++)
  {
    SystemComponents.MolFraction[j] /= tot;
    printf("Component [%zu] (%s), Mol Fraction: %.5f\n", j, SystemComponents.MoleculeName[j].c_str(), SystemComponents.MolFraction[j]);
  }
  printf("=================================================\n");
}
