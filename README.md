# Molecular dynamics simulation of nitrogen diffusion in iron and iron nitrides using *ab initio* data trained machine learning potentials

This repository contains the data, trained model, and example workflows **related to the paper**:  
**“Molecular dynamics simulation of nitrogen diffusion in iron and iron nitrides using ab initio data trained machine learning potentials.”** 

It provides the **DFT/AIMD training dataset**, a **trained DeepMD (DeepMD-kit) machine-learning interatomic potential (MLP[All])**, and **reproducible example workflows** used in the accompanying study on nitrogen diffusion in iron and iron nitrides. 

Contents (high level):
- `diffusion_fexny_vdw11/`: DFT/AIMD snapshots (train/test splits) for 10 crystalline Fe–N materials.
- `01.deepmd_train_example/`: example DeepMD-kit training inputs/outputs and the exported model (`deepmd_potential.pb`).
- `02.lammps_run_example/`: example LAMMPS + DeepMD run for equilibration and diffusion/MSD analysis.


## Dataset description: `diffusion_fexny_vdw11/`

This repository contains the DFT/AIMD training and test datasets used to develop my Fe–N machine-learning interatomic potential (MLP) for nitrogen diffusion in iron and iron nitrides.

### Top-level organization
- **`diffusion_fexny_vdw11/`** is the dataset root (the suffix `vdw11` is my internal label for the DFT setup; full computational parameters and snapshot sampling details are documented in the main text and Supplementary Information).
- Under the root, there are **10 crystalline materials**. Each material folder name encodes an index, a chemical formula (or shorthand), and the **Materials Project (MP) ID** of the reference crystal structure (e.g., `01_fe_mp13`, `03_fe_mp150`, …).

### Train/test split
Inside each material folder, I provide:
- **`train/`**: training split
- **`test/`**: test split

`train/` and `test/` use the same condition names. They are implemented as convenient links (symlinks) to the corresponding condition-specific directories, so readers can access either the curated split (`train/` / `test/`) or the original condition folders.

### Data format
All snapshots are exported using **dpdata** in the **DeepMD-kit NumPy format (`deepmd/npy`)**, and are stored under paths like:
- `dpmd_npy_train*/set.000/` and `dpmd_npy_test*/set.000/`

The `set.000/` directory is a standard DeepMD shard container with NumPy arrays for atomic configurations and labels (e.g., coordinates, cell/box, energies, forces; and optionally virials), plus atom-type metadata. This makes the dataset directly loadable using `dpdata` and compatible with DeepMD workflows.

Reference implementation (dpdata):
- https://github.com/deepmodeling/dpdata

### Condition naming convention (shared by `train/` and `test/`)
The condition labels are designed to map directly to the descriptions in the paper/SI (parameters, sampling strategy, and snapshot counts are reported there). Here I summarize the meaning of each label:

#### (1) Equilibrium AIMD snapshots at different temperatures
- **`T500eq`**: equilibrium AIMD snapshots at **500 K**
- **`T1000eq`**: equilibrium AIMD snapshots at **1000 K**
- **`T1500eq`**: equilibrium AIMD snapshots at **1500 K**

#### (2) Stoichiometry perturbations at 1000 K (adding/removing N)
- **`T1000_Np1`**, **`T1000_Np2`**, **`T1000_Np4`**: AIMD snapshots at **1000 K** with **+1, +2, or +4 nitrogen atoms** introduced into the reference lattice
- **`T1000_Nm2`**: AIMD snapshots at **1000 K** with **2 nitrogen atoms removed** from the reference lattice

#### (3) Volume/strain perturbations around equilibrium at 1000 K
- **`bmeq_095_T1000`**: AIMD snapshots at **1000 K** with lattice constants **compressed to 0.95×** their equilibrium value
- **`bmeq_105_T1000`** or **`bmeq_103_T1000`**: AIMD snapshots at **1000 K** with lattice constants **expanded to 1.05×** or **1.03×** their equilibrium value

### Practical navigation
For a given material and condition, the curated split can be accessed as:
- `diffusion_fexny_vdw11/<material>/train/<condition>/`
- `diffusion_fexny_vdw11/<material>/test/<condition>/`

The corresponding raw condition directories (e.g., `md_T500/`, `nx/`, `md_bmeq_bulk/`) contain the same snapshots with provenance-oriented organization.

### Example: reading a test split and printing basic statistics with dpdata (`read_data_example.ipynb`)

To help readers quickly verify and explore the released datasets, I provide a minimal Python example (`read_data_example.ipynb`) that loads **one `test/` condition** from any material folder using **dpdata** (format: `deepmd/npy`) and prints basic statistics. The script reports:

- number of frames and composition (atom types and counts),
- dataset-level energy statistics (min/max/mean/std),
- dataset-level force magnitude statistics (min/max/mean/std),
- and a sampled per-frame table of energy and force-norm summary values.

This is intended as a lightweight sanity-check and a starting point for downstream analyses (e.g., filtering, visualization, or re-splitting). The dataset path can be changed to any entry under  
`diffusion_fexny_vdw11/<material>/test/<condition>/`, such as `T500eq`, `T1000eq`, `T1500eq`, `T1000_Np*`, `T1000_Nm*`, or `bmeq_*_T1000`.
## deepmd-kit training example (`01.deepmd_train_example`)

This folder provides a minimal **DeepMD-kit training workflow example** corresponding to the dataset release. It contains the key input, output logs, and the exported model file (`deepmd_potential.pb`) that can be used directly in downstream MD (e.g., the LAMMPS example in `02.lammps_run_example/`).

### Files

- **`input.json`**  
  The **DeepMD-kit training configuration** used to run `dp train`.  
  Typically defines: dataset paths (train/validation), descriptor/network architecture, learning rate schedule, loss weights (energy/force/virial), batch sizes, number of steps, random seeds, and other runtime options.  
  This is the primary file to reproduce the training procedure.

- **`stdlog`**  
  The **standard output log** captured during the training run (i.e., console output).  
  Useful for quickly checking run progress, step-wise loss values, learning rate changes, and any warnings/errors emitted during training.

- **`lcurve.out`**  
  The **learning-curve record** written by DeepMD-kit, containing the evolution of training/validation metrics over optimization steps (e.g., RMSE for energy/forces/virials).  
  This file is commonly used to plot convergence and diagnose under/over-fitting.

- **`out.json`**  
  The **training summary produced by DeepMD-kit**.  
  This is helpful for documenting the exact settings that were executed (including defaults DeepMD fills in) and for programmatic extraction of final errors.

- **`sub.sh`**  
  A **job submission script** (HPC example) to launch training on a cluster environment.
  Typically includes environment setup (modules/conda), resource requests (CPU/GPU), and the training command (e.g., `dp --tf train input.json > ./stdlog 2>&1` and `dp --tf freeze -o "deepmd_potential.pb"`). This is to show the command that I used to initiate the training of MLP and freeze the MLP.

- **`deepmd_potential.pb`**  
  The **exported trained DeepMD model**.
  This is the deployable potential file used for inference in MD engines (e.g., `pair_style deepmd .../deepmd_potential.pb` in LAMMPS).  
  In this release, it corespondes to the reported `MLP[All]` described in the paper.




## LAMMPS run example (`02.lammps_run_example/`)

In addition to the training/test datasets, I provide a minimal **LAMMPS + DeepMD** run example to help other researchers reproduce a typical MD workflow using the released potential. This folder is intended as a practical **“test”**: if your LAMMPS + DeepMD-kit environment is correctly configured, you should be able to run the input and obtain thermodynamic outputs and mean-square-displacement (MSD) trajectories.

### Files
- `CONTCAR_07_fe12N5_mp27908_sup223_oct1234Np4_334`: example starting structure (read by LAMMPS via `read_data`).
- `input`: input files used for the lammps run.
- `sub.lammps.sh`: job submission script (cluster/HPC example).
- `stdout.log`, `log.lammps`: example standard outputs produced by LAMMPS.
- `msd.data`: example MSD time series output.

**What the LAMMPS input does**
1. **Loads the structure** and assigns a DeepMD potential  
   - `read_data ${input_stu}` reads the example configuration.  
   - `pair_style deepmd ../01.deepmd_train_example/deepmd_potential.pb` points to the trained DeepMD model file distributed in this release.
2. **Defines species groups for MSD analysis**  
   - `group Fes type 1`, `group Ns type 2` separate Fe and N atoms (type-1 = Fe, type-2 = N in the data file).
3. **Equilibrates the cell at the target temperature** (default: **1000 K**)  
   - Uses `fix npt` with a triclinic barostat (`tri`) at ~1 atm (`1.01325` bar) to relax volume/shape before diffusion sampling.
4. **(Optional) high-temperature disordering / annealing block**  
   - Controlled by `variable skip_hiTdiso equal false`.  
   - When enabled, the script ramps **1000 → 1500 K**, holds at 1500 K, then cools back to 1000 K, dumping a trajectory (`nvt_hiTdiso_traj.dat`).  
   - Set `skip_hiTdiso=true` to bypass this block.
5. **Runs long NVT diffusion sampling and outputs MSD**  
   - Resets timestep to 0, then runs NVT at the target temperature.  
   - Computes MSD for Fe and N separately with center-of-mass removal (`compute ... msd com yes`).  
   - Writes an averaged time series to `msd.data` via `fix ave/time`, and dumps a diffusion trajectory (`diff_traj.dat`).

### Outputs you can expect
- Thermodynamic log (`log.lammps`) including `step`, `temp`, `pe`, `ke`, `press`, `vol`, and cell parameters.
- Relaxed structures / snapshots (`npt.data`, `npt.atom`, optional annealing dumps).
- Diffusion trajectory (`diff_traj.dat`) and MSD time series (`msd.data`) for both N and Fe, which can be post-processed to estimate diffusion coefficients.


## License

- **Code, scripts, examples, and trained model** (e.g., `01.deepmd_train_example/`, `02.lammps_run_example/`, and `01.deepmd_train_example/deepmd_potential.pb`): Apache License 2.0 (see `LICENSE`).
- **DFT/AIMD dataset** (`diffusion_fexny_vdw11/**`): Creative Commons Attribution 4.0 International (CC BY 4.0) (see `LICENSE-DATA`).



> **Note on generative AI use:** Portions of the dataset documentation (folder/file descriptions and wording) were drafted with the assistance of a generative AI tool and then reviewed/edited by the authors for accuracy.
