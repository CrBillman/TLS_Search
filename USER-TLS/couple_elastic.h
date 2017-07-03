/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/*--Coded by Jonathan Trinastic, University of Florida 1/21/2016--------- */

#ifdef COMMAND_CLASS

CommandStyle(couple_elastic,CoupleElastic)

#else

#ifndef LMP_COUPLE_ELASTIC_H
#define LMP_COUPLE_ELASTIC_H

#include "pointers.h"

namespace LAMMPS_NS {

  class CoupleElastic : protected Pointers {
  public:
    CoupleElastic(class LAMMPS *);
    ~CoupleElastic();
    void command(int, char **);
    double memory_usage();
    void allocate(int nStrain);
   
  protected:
    //char *config1name, *config2name; // configuration names
    FILE * of1; // output file of raw data
    FILE * of2; // output file of final data for internal friction
    FILE * of3; // output file of elastic constants
    char *fitType; // type of function fitting
    double eps; // max scaled strain percent
    double **oTLS; // TLS minimum config
    double *oLat; // original lattice prior to strains
    double **strainTLS; // stores lattice vector from each strain
    double ***energyTLS; // stores energy min vs strain for each TLS
    double ****stressTLS; // stores stress tensor for each TLS/strain
    double **deltaTLS; // stores asymmetry vs strain for each TLS
    double *cc; // coupling constants for each strain direction
    double ***ec; // elastic constants for each TLS
    double **em; // elastic moduli for each TLS
    int pair_compute_flag; // 0 if pair->compute is skipped
    int kspace_compute_flag; // 0 if kspace->compute is skipped
    int pflag;
    bool relaxFlag;

    class Compute *pressure;

    int LoadPositions(char * num, bool rFlag); // load Fix_Store positions
    void InitPressCompute(); // Create pressure compute
    void ApplyStrain(char *dir, char *strain); // apply strain
    void UndoStrain(double * lattice); // return to pre-strain box
    double CallMinimize(); // call minimize command
    void CalcStressTensor(); // calculate pressure elements
    void FitLinearCC(int strain, double ** xInput, double ** yInput, double * gamma); // linear cc fitting
    void FitLinearEC(int strain, double ** xInput, double **** yInput, double *** elastic); // linear cc fitting
    void CalcElasticMod(double *** InputEC, double ** OutputEM); // elastic moduli calculation
    void CopyBoxToLat(double * latArray);
    void CopyLatToBox(double * latArray);
    void CopyAtoms(double** copyArray, double** templateArray);
    void MappedCopyAtoms(double** copyArray, double** templateArray);
    void ConvertDoubleToChar(double doubleInput, char * charOutput);
    void OpenOutputData();
    void OpenOutputFitting();
    void OpenOutputElastic();
    void force_clear(); // Clear current force array
    void InitialRelaxation();
    void InitialMinimization();

  };
 
}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

W: More than one couple/elastic command
*/
