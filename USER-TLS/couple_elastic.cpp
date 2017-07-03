/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* Coded by Jonathan Trinastic, University of Florida 1/21/2016---------- */

#include <iostream>
#include <string.h>
#include <cstring>
#include <sstream>
#include <cmath>
#include "couple_elastic.h"
#include "run.h"
#include "integrate.h"
#include "output.h"
#include "input.h"
#include "min.h"
#include "minimize.h"
#include "fix.h"
#include "pointers.h"
#include "domain.h"
#include "lattice.h"
#include "atom.h"
#include "atom_vec.h"
#include "update.h"
#include "modify.h"
#include "neighbor.h"
#include "comm.h"
#include "force.h"
#include "pair.h"
#include "bond.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"
#include "kspace.h"
#include "memory.h"
#include "error.h"
#include "stdlib.h"
#include "string.h"
#include "mpi.h"
#include "math.h"
#include "fix_store.h"
#include "fix_store_state.h"
#include "fix_store_lat.h"
#include "change_box.h"
#include "thermo.h"
#include "compute.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

CoupleElastic::CoupleElastic(LAMMPS *lmp) : Pointers(lmp)
{}

/* ---------------------------------------------------------------------- */

CoupleElastic::~CoupleElastic()
{
  memory->destroy(oTLS);
  memory->destroy(oLat);
  memory->destroy(strainTLS);
  memory->destroy(energyTLS);
  memory->destroy(stressTLS);
  memory->destroy(deltaTLS);
  memory->destroy(cc);
  memory->destroy(ec);
  memory->destroy(em);


}

/* ---------------------------------------------------------------------- */

void CoupleElastic::command(int narg, char **arg)
{

  /* read in all input parameters: */
//  std::cout << "Got here 88" << std::endl;
  // code currently looks for exactly two parameters
  if (narg < 2) error->all(FLERR,"Illegal couple_elastic command");

  // code requires simulation box to apply strain
  if (domain->box_exist == 0)
    error->all(FLERR,"couple_elastic command requires simulation box be defined");

  // code requires triclinic cell for xy, xz, yz strains
  if (domain->triclinic != 1)
    error->all(FLERR,"couple_elastic command requires triclinic cell");

  eps = atof(arg[0]); /* max scaled strain percent*/
//  std::cout << "Got here 101" << std::endl;
  int n = strlen(arg[1]) + 1;
  //fitType = new char[n]; /* fitting type: linear is only current option */
  char fitType[n];
  strcpy(fitType,arg[1]);

  relaxFlag = false;
  if(narg == 3 && !strcmp(arg[2],"relax")) relaxFlag = true;


  /* initialize arrays and values */
  oTLS = NULL;
  oLat = NULL;
  strainTLS = NULL;
  energyTLS = NULL;
  stressTLS = NULL;
  deltaTLS = NULL;
  cc = NULL;
  ec = NULL;
  em = NULL;
  char * strainDir;
  char * numTLS;
  int numStrain = 3; /* number of strain intervals in one direction */
  numTLS = "1";
  allocate(numStrain);
  if( LoadPositions(numTLS, false) ) return;
//  std::cout << "Got here 126" << std::endl;
  //allocate(numStrain); /* allocate memory for arrays */

  /* Open files to write output */
  OpenOutputData(); /* output file with raw strain, asymmetry, stres data */
  OpenOutputElastic(); /* output file with elastic constant matrix for each TLS */
  OpenOutputFitting(); /* output file with final coupling constants and elastic moduli */

  /* Set up map if none exists (from velocity.cpp line 248) */
  int mapflag = 0;
  if (atom->map_style == 0) {
    mapflag = 1;
    atom->map_init();
    atom->map_set();
  }

  /* Initialize pressure compute to use later */
  InitPressCompute();
  double dNumStrain = double(numStrain);
  double strainMag;
  char * chStrainMag = new char[10];
//  std::cout << "Got here 155" << std::endl;
  /* Outer loop over each minimum configuration of TLS */
  // Must do one at a time to load positions correctly
  for (int i = 0; i < 2; i++)
    {
      // Identify TLS for loading positions
      if (i == 0){numTLS = "1";}
      if (i == 1){numTLS = "2";}

      // Load atomic positions of TLS
      if( LoadPositions(numTLS, relaxFlag) ) return;

    //  std::cout << "Got here 2" << std::endl;

    //  std::cout << "Got here 4" << std::endl;

      // Load box parameters
      /* TO DO */
        
      // Store lattice of current TLS
      //CopyBoxToLat(oLat);  

      // Apply strain and calculate changes in energy and stress tensor
      for (int j =0; j < 6; j++)
	{
	  if (j == 0) {strainDir = "x";}
	  if (j == 1) {strainDir = "y";}
	  if (j == 2) {strainDir = "z";}
	  if (j == 3) {strainDir = "xy";}
	  if (j == 4) {strainDir = "xz";}
	  if (j == 5) {strainDir = "yz";}

	  // Loop over strain intervals
	  for (int k = 0; k < (2 * numStrain + 1); k++)
	    {
	      // Load original TLS positions
	      LoadPositions(numTLS, false);

	      // Calculate strain amount for given interval step
	      // Starts from maximum negative strain, works up
	      //strainMag = (2 - eps) + k * ((eps - 1) / dNumStrain);
	      strainMag = 1.0 - eps + k * eps / dNumStrain;
	      if(comm->me == 0) 
	      {
		fprintf(screen, "CR_UPDATE- %f\t%f\t%f\t%f\t%f\t%f\n", domain->h[0], domain->h[1], domain->h[2], domain->h[3], domain->h[4], domain->h[5]);
		std::cout << "For k = " << k << ", applying strain of " << strainMag << std::endl;
	      }
	      ConvertDoubleToChar(strainMag, chStrainMag);

	      // Apply strain and store in strain array
	      ApplyStrain(strainDir, chStrainMag);

	      // Save strain magnitude to strain array
	      strainTLS[j][k] = strainMag - 1;

	      // Minimize and store energy
	      energyTLS[i][j][k] = CallMinimize();

	      // Calculate and store stress tensor
	      CalcStressTensor();
	      stressTLS[i][j][k][0] = pressure->vector[0];
	      stressTLS[i][j][k][1] = pressure->vector[1];
	      stressTLS[i][j][k][2] = pressure->vector[2];
	      stressTLS[i][j][k][3] = pressure->vector[3];
	      stressTLS[i][j][k][4] = pressure->vector[4];
	      stressTLS[i][j][k][5] = pressure->vector[5];

	      // Undo strain
	      UndoStrain(oLat);

	    } 
	}

    } // end of TLS1/2 loop

  // Reset to original TLS1 positions
  LoadPositions("1", false);

  /* Calculate asymmetry */
  for (int j = 0; j < 6; j++)
    {
      for (int k = 0; k < (2 * numStrain + 1); k++)
	{
	  deltaTLS[j][k] = energyTLS[1][j][k] - energyTLS[0][j][k];
	}
    }
  
  /* Function fitting */
  //Linear fitting
  if(strcmp(fitType, "linear") == 0)
    {
      FitLinearCC(numStrain, strainTLS, deltaTLS, cc);
      FitLinearEC(numStrain, strainTLS, stressTLS, ec);
    }
  //Fitting selected is not supported
  else
    {
      error->all(FLERR,"couple_elastic command only supports linear fitting");
    }

  /* Calculate elastic moduli */
  CalcElasticMod(ec, em);

  /* Output all data to files */
  if(comm->me == 0)
    {
      // Header for raw data file
      fprintf(of1, "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s \n", "Strain", "Energy1", "Energy2", "Asymmetry", "Pxx1", "Pyy1", "Pzz1", "Pxy1", "Pxz1", "Pyz1", "Pxx2", "Pyy2", "Pzz2", "Pxy2", "Pxz2", "Pyz2");
      // Header for fitting file
      fprintf(of2, "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s \n", "GammaXX", "GammaYY", "GammaZZ", "GammaXY", "GammaXZ", "GammaYZ", "B1", "G1", "Y1", "P1", "B2", "G2", "Y2", "P2");
    }

  // Output to raw data file
  for (int j =0; j < 6; j++)
    {
      for (int k = 0; k < (2 * numStrain + 1); k++)
	{
	  // Calculate asymmetry
	  deltaTLS[j][k] = energyTLS[1][j][k] - energyTLS[0][j][k];
	  
	  // Output to raw data file
	  if(comm->me == 0)
	    {
	      // Output energy, asymmetry, and stress tensor for given strain for each TLS (asymmetry will be one value only)
	      fprintf(of1, "%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f \n", strainTLS[j][k], energyTLS[0][j][k], energyTLS[1][j][k], deltaTLS[j][k], stressTLS[0][j][k][0], stressTLS[0][j][k][1], stressTLS[0][j][k][2], stressTLS[0][j][k][3], stressTLS[0][j][k][4], stressTLS[0][j][k][5], stressTLS[1][j][k][0], stressTLS[1][j][k][1], stressTLS[1][j][k][2], stressTLS[1][j][k][3], stressTLS[1][j][k][4], stressTLS[1][j][k][5]); 
	    }
	}
      if(comm->me ==0) fprintf(of1, "\n"); 
    }

  // Output to fitting file
  if (comm->me == 0)
    {
      fprintf(of2, "%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f \n", cc[0], cc[1], cc[2], cc[3], cc[4], cc[5], em[0][0], em[0][1], em[0][2], em[0][3], em[1][0], em[1][1], em[1][2], em[1][3]); 
    }

  // Output to elastic constants file
  if (comm->me == 0)
    {
      for (int i = 0; i < 2; i++)
	{
	  for (int j = 0; j < 6; j++)
	    {
	      fprintf(of3, "%f\t%f\t%f\t%f\t%f\t%f \n", ec[i][j][0], ec[i][j][1], ec[i][j][2], ec[i][j][3], ec[i][j][4], ec[i][j][5]); 
	    }
	  fprintf(of3, " \n");
	}
    }

  /* Clean-up */

  // Close output files
  if(comm->me==0) fclose(of1);
  if(comm->me==0) fclose(of2);  
  if(comm->me==0) fclose(of3);  

  // Delete map
  if (atom->map_style!=0) {
    atom->map_delete();
    atom->map_style = 0;
  }

  // Delete pressure compute used just for these calculations
  modify->delete_compute("cc_ec");

  // Turn on fix_box_relax if originally on before these calculations
  /* TO DO */

  //Delete local variables
  //delete [] fitType;
  delete [] chStrainMag;

}

/* ---------------------------------------------------------------------- */
int CoupleElastic::LoadPositions(char *num, bool rFlag)
{
//  std::cout << "Got here 324" << std::endl;
  double **x = atom->x;
  // Create name of FixStore to look for
  int l = strlen("TLS") + 1;
  char * TLSname = new char[l];
  strcpy(TLSname, "TLS");
  strcat(TLSname, num);
//  std::cout << "Got here 331" << std::endl;
  //Get the labels for the fixes for the TLS atom positions.
  int iTLS = modify->find_fix(TLSname);

//  std::cout << "Got here 336" << std::endl;
  //If there are no corresponding fixes, returns -1 to flag the error.
  if(iTLS<0)
  {
     if(comm->me == 0) fprintf(screen, "No FixStore for %s, exitting couple_elastic\n", TLSname);
     return 1;
  }
//  std::cout << "Got here 343" << std::endl;

  //Creates a fix according to the stored fix
  FixStore * TLS = (FixStore *) modify->fix[iTLS];
  CopyAtoms(x, TLS->astore);
//  std::cout << "Got here 348" << std::endl;

  l = strlen("TLSLat") + 1;
  
  delete [] TLSname;
  TLSname = new char[l];
  strcpy(TLSname, "TLSLat");
  strcat(TLSname, num);

//  std::cout << "Got here 357" << std::endl;
  iTLS = modify->find_fix(TLSname);
  delete [] TLSname;
  if(iTLS < 0)
  {
    return -1;
  }
 // std::cout << "Got here 364" << std::endl;
  FixStoreLat * TLSlat = (FixStoreLat *) modify->fix[iTLS];

//Only for debugging
/*
  for(int i = 0; i < atom->nlocal; i++)
  {
    std::cout << atom->tag[i] << "  " << atom->x[i][0] << "  " << atom->x[i][1] << "  " << atom->x[i][2] << std::endl;
  }
*/

  if(rFlag)
  {
    //CopyLatToBox(TLSlat->vstore);
    UndoStrain(TLSlat->vstore);
    InitialRelaxation();
    //InitialMinimization();
    CopyBoxToLat(TLSlat->vstore);
    CopyAtoms(TLS->astore, x);
  }

 // std::cout << "Got here 376" << std::endl;
  //This is the last place that the code gets to
  for(int i = 0; i < 9; i++)
  {
    oLat[i] = TLSlat->vstore[i];
  }

//  std::cout << "Got here 382" << std::endl;
  return 0;
}

/* ---------------------------------------------------------------------- */
/* -- Function to apply strain in each direction --*/
/* -- Defaults: all, scale style, remap atomic ooord, box units -- */
void CoupleElastic::ApplyStrain(char * dir, char * strain)
{
  // Longitudinal strains:
  if ((dir == "x") || (dir == "y") || (dir == "z"))
    {
      char **newarg = new char*[7];
      // Create arguments for change_box
      newarg[0] = (char *) "all";
      newarg[1] = dir;
      newarg[2] = (char *) "scale";
      newarg[3] = strain;
      newarg[4] = "remap";
      newarg[5] = (char *) "units";
      newarg[6] = (char *) "box";
      
      // Apply change_box command
      ChangeBox* bChange = new ChangeBox(lmp);
      bChange->command(7, newarg);
      delete bChange;
      delete [] newarg;
    }

  // Transverse strains:
  else if ((dir == "xy") || (dir == "xz") || (dir == "yz"))
    {
      char **newarg = new char*[7];
      
      // Create arguments for change_box
      // Note: for transverse strains, must convert percent strain to 
      // raw magnitude //
      double strainT;
      double dStrain = atof(strain);
      if ((dir == "xy")) {strainT = 0.5*((dStrain - 1)*domain->prd[0] + (dStrain - 1)*domain->prd[1]);}
      if ((dir == "xz")) {strainT = 0.5*((dStrain - 1)*domain->prd[0] + (dStrain - 1)*domain->prd[2]);}
      if ((dir == "yz")) {strainT = 0.5*((dStrain - 1)*domain->prd[1] + (dStrain - 1)*domain->prd[2]);}
      char chStrainT[10];
      ConvertDoubleToChar(strainT, chStrainT);

      // Create arguments for change_box command
      newarg[0] = (char *) "all";
      newarg[1] = dir;
      newarg[2] = (char *) "delta";
      newarg[3] = chStrainT;
      newarg[4] = "remap";
      newarg[5] = (char *) "units";
      newarg[6] = (char *) "box";
      
      // Apply change_box command
      ChangeBox* bChange = new ChangeBox(lmp);
      bChange->command(7, newarg);
      delete bChange;
      delete [] newarg;
      //delete [] chStrainT;
    }

}

/* -- Function to undo strain after energy, stress tensor calculation --*/
void CoupleElastic::UndoStrain(double * lattice)
{
  // Copy original lattice to boxlo, boxhi, and tilt factors
  CopyLatToBox(lattice);

  // Initialize, then set global and local box to reset original
  domain->set_initial_box();
  domain->set_global_box();
  domain->set_local_box();
}

/* ---------------------------------------------------------------------- */

double CoupleElastic::CallMinimize()
{
  char **newarg = new char*[4];
  newarg[0] = (char *) "0";
  newarg[1] = (char *) "1.0e-6";
  newarg[2] = (char *) "10000";
  newarg[3] = (char *) "10000";
  Minimize* rMin = new Minimize(lmp);
  rMin->command(4, newarg);

  delete rMin;
  delete [] newarg;
  return update->minimize->efinal;

}

/* ---------------------------------------------------------------------- */

void CoupleElastic::InitPressCompute()
{
  // Create pressure compute
  // Default: virial to not take temp into account
  char **newarg = new char*[5];
  newarg[0] = (char *) "cc_ec";
  newarg[1] = (char *) "all";
  newarg[2] = (char *) "pressure";
  newarg[3] = (char *) "NULL";
  newarg[4] = (char *) "virial";
  modify->add_compute(5,newarg);

  int icompute = modify->find_compute("cc_ec");
  pressure = modify->compute[icompute];

  delete [] newarg;
}

/* ---------------------------------------------------------------------- */

void CoupleElastic::CalcStressTensor()
{
  pressure->compute_vector();

}

/* ---------------------------------------------------------------------- */

void CoupleElastic::FitLinearCC(int strain, double ** xInput, double ** yInput, double * gamma)
{
  for (int j = 0; j < 6; j++)
    {
      gamma[j] = 0.0;
      int counter = 1;
      double sumX = 0.0;
      double sumY = 0.0; 
      double sumXY = 0.0;
      double sumX2 = 0.0;
      for (int k = 0; k < (2 * strain + 1); k++)
	{
	  counter = counter + 1;
	  sumX = sumX + xInput[j][k];
	  sumY = sumY + yInput[j][k];
	  sumXY = sumXY + xInput[j][k]*yInput[j][k];
	  sumX2 = sumX2 + xInput[j][k]*xInput[j][k];
	}
      gamma[j] = fabs(0.5 * ( (counter * sumXY - sumX * sumY) / (counter * sumX2 - pow(sumX,2))));
    }
}

/* ---------------------------------------------------------------------- */

void CoupleElastic::FitLinearEC(int strain, double ** xInput, double **** yInput, double *** elastic)
{
  //Outer loop over TLS number
  for (int i = 0; i < 2; i++)
    {
      // Second loop over strain direction
      for (int j = 0; j < 6; j++)
	{
	  // Third loop over stress direction
	  for (int m = 0; m < 6; m++)
	    {
	      int counter = 1;
	      double sumX = 0.0;
	      double sumY = 0.0; 
	      double sumXY = 0.0;
	      double sumX2 = 0.0;
	      for (int k = 0; k < (2 * strain + 1); k++)
		{
		  counter = counter + 1;
		  sumX = sumX + xInput[j][k];
		  sumY = sumY + yInput[i][j][k][m];
		  sumXY = sumXY + xInput[j][k]*yInput[i][j][k][m];
		  sumX2 = sumX2 + xInput[j][k]*xInput[j][k];
		}
	      elastic[i][m][j] = fabs((((counter * sumXY) - (sumX * sumY)) / (counter * sumX2 - pow(sumX,2)))); 
	    }
	}
    }
}

/* ---------------------------------------------------------------------- */

void CoupleElastic::CalcElasticMod(double *** inputEC, double ** outputEM)
{
  // Calculate elastic moduli for each TLS
  for (int i = 0; i < 2; i++)
    {
      // [0] is Bulk, [1] is Shear, [2] is Young's, [3] is Poisson's ratio
      // Note: calculated using Voight formula
      // Note: Formula for orthorhombic cell - must change for triclinic
      outputEM[i][0] = (1.0/9.0) * (inputEC[i][0][0] + inputEC[i][1][1] + inputEC[i][2][2]) + (2.0/9.0) * (0.5*(inputEC[i][0][1]+inputEC[i][1][0]) + 0.5*(inputEC[i][0][2]+inputEC[i][2][0]) + 0.5*(inputEC[i][1][2]+inputEC[i][2][1]));
      outputEM[i][1] = (1.0/15.0) * (inputEC[i][0][0] + inputEC[i][1][1] + inputEC[i][2][2] - 0.5*(inputEC[i][0][1]+inputEC[i][1][0]) - 0.5*(inputEC[i][0][2]+inputEC[i][2][0]) - 0.5*(inputEC[i][1][2]+inputEC[i][2][1])) + (1.0/5.0) * (inputEC[i][3][3] + inputEC[i][4][4] + inputEC[i][5][5]);
      outputEM[i][2] = (9.0 * outputEM[i][0] * outputEM[i][1]) / (3.0 * outputEM[i][0] + outputEM[i][1]);
      outputEM[i][3] = (3.0 * outputEM[i][0] - 2.0 * outputEM[i][1]) / (2.0 * (3.0 * outputEM[i][0] + outputEM[i][1]));
    }

}


/* ---------------------------------------------------------------------- */

void CoupleElastic::MappedCopyAtoms(double** copyArray, double** templateArray)
{
  int m;
  for(int i = 1; i <= atom->natoms; i++)
    {
      m = atom->map(i);
      if(m >= 0 && m < atom->nlocal)
	{
	  for(int j = 0; j < domain->dimension; j++)
	    {
	      copyArray[m][j] = templateArray[m][j];
	    }
	}
    }
}

/* ---------------------------------------------------------------------- */

void CoupleElastic::CopyAtoms(double** copyArray, double** templateArray)
{
  for(int i=0; i < atom->nlocal; i++)
    {
      for(int j = 0; j < domain->dimension; j++)
	{
	  copyArray[i][j] = templateArray[i][j];
	}
    }
}

/* ---------------------------------------------------------------------- */

void CoupleElastic::CopyBoxToLat(double * latArray)
{
  latArray[0] = domain->boxlo[0];
  latArray[1] = domain->boxlo[1];
  latArray[2] = domain->boxlo[2];
  latArray[3] = domain->boxhi[0];
  latArray[4] = domain->boxhi[1];
  latArray[5] = domain->boxhi[2];
  latArray[6] = domain->xy;
  latArray[7] = domain->xz;
  latArray[8] = domain->yz;
}

/* ---------------------------------------------------------------------- */

void CoupleElastic::CopyLatToBox(double * latArray)
{
  domain->boxlo[0] = latArray[0];
  domain->boxlo[1] = latArray[1];
  domain->boxlo[2] = latArray[2];
  domain->boxhi[0] = latArray[3];
  domain->boxhi[1] = latArray[4];
  domain->boxhi[2] = latArray[5];
  domain->xy = latArray[6];
  domain->xz = latArray[7];
  domain->yz = latArray[8];
}

/* ---------------------------------------------------------------------- */

void CoupleElastic::ConvertDoubleToChar(double doubleInput, char * charOutput)
{
  sprintf(charOutput, "%1f", doubleInput);
}

/* ---------------------------------------------------------------------- */

void CoupleElastic::OpenOutputData()
{
  std::string strFile = "cc_ec_raw_data.dump";
  char *charFile = new char[20];
  std::strcpy(charFile, strFile.c_str());
  of1 = fopen(charFile, "a");
  delete [] charFile;
  return;
}

/* ---------------------------------------------------------------------- */

void CoupleElastic::OpenOutputElastic()
{
  std::string strFile = "cc_ec_elastic.dump";
  char *charFile = new char[20];
  std::strcpy(charFile, strFile.c_str());
  of3 = fopen(charFile, "a");
  delete [] charFile;
  return;
}

/* ---------------------------------------------------------------------- */

void CoupleElastic::OpenOutputFitting()
{
  std::string strFile = "cc_ec_final.dump";
  char *charFile = new char[20];
  std::strcpy(charFile, strFile.c_str());
  of2 = fopen(charFile, "a");
  delete [] charFile;
  return;
}

/* ---------------------------------------------------------------------- */

void CoupleElastic::InitialRelaxation()
{
//  std::cout << "In IR" << std::endl;
  int Steps = 1000;
    int maxLoops = 10;
  int me;
  int alphaCounter = 0;
  char cFSteps[10];
  //std::cout << "Finished initializations" << std::endl;
  MPI_Comm_rank(world,&me);
  char **newarg = new char*[4];
  newarg[0] = (char *) "0.0";
  newarg[1] = (char *) "1.0e-6";
  newarg[2] = (char *) "1000";
  newarg[3] = (char *) "10000";
  //std::cout << "Created newarg" << std::endl;

  char **brArg = new char*[7];
  brArg[0] = (char *) "SearchBoxRelax";
  brArg[1] = (char *) "all";
  brArg[2] = (char *) "box/relax";
  brArg[3] = (char *) "tri";
  brArg[4] = (char *) "0.0";
  brArg[5] = (char *) "vmax";
  brArg[6] = (char *) "0.01";
  //std::cout << "Created brArg" << std::endl;

  char **frArg = new char*[6];
  frArg[0] = (char *) "SearchFreeze";
  frArg[1] = (char *) "all";
  frArg[2] = (char *) "setforce";
  frArg[3] = (char *) "0.0";
  frArg[4] = (char *) "0.0";
  frArg[5] = (char *) "0.0";
  //std::cout << "Created frArg" << std::endl;

  if(comm->me == 0) fprintf(screen, "CR_UPDATE- Starting Relaxation of Box\n");
  if(comm->me == 0) fprintf(screen, "CR_UPDATE- %f\t%f\t%f\t%f\t%f\t%f\n", domain->h[0], domain->h[1], domain->h[2], domain->h[3], domain->h[4], domain->h[5]);
  if(comm->me == 0) std::cout << "Printed out h-matrix stuff" << std::endl;
  for(int i = 0; i < maxLoops; i++)
  {
    if(comm->me == 0) fprintf(screen, "CR_UPDATE- BR %d\n", i);
    if(comm->me == 0) fprintf(screen, "CR_UPDATE- %f\t%f\t%f\t%f\t%f\t%f\n", domain->h[0], domain->h[1], domain->h[2], domain->h[3], domain->h[4], domain->h[5]);
    modify->add_fix(6, frArg, 1);
    modify->add_fix(7, brArg, 1);
    Minimize* rMinBox = new Minimize(lmp);
    rMinBox->command(4, newarg);
  //  std::cout << "did box min" << std::endl;
    delete rMinBox;

    modify->delete_fix(brArg[0]);
    modify->delete_fix(frArg[0]);
    if(comm->me == 0) fprintf(screen, "CR_UPDATE- AR %d\n", i);
    if(comm->me == 0) fprintf(screen, "CR_UPDATE- %f\t%f\t%f\t%f\t%f\t%f\n", domain->h[0], domain->h[1], domain->h[2], domain->h[3], domain->h[4], domain->h[5]);
    Minimize* rMinAtoms = new Minimize(lmp);
    rMinAtoms->command(4, newarg);
    delete rMinAtoms;
      if(update->minimize->stop_condition == 3) break;
  }

  if(comm->me == 0) fprintf(screen, "Xlo is now %f\n",  domain->boxlo[0]);

  CopyBoxToLat(oLat);
  delete [] newarg;
  delete [] brArg;
  delete [] frArg;

  return;
}

/* ----------------------------------------------------------------------


   ------------------------------------------------------------------------- */

void CoupleElastic::InitialMinimization()
{
//  std::cout << "In IR" << std::endl;
  int Steps = 1000;
  int maxLoops = 30;
  int me;
  int alphaCounter = 0;
  char cFSteps[10];
  double minEnergy = 1e10;
  double currEnergy;
  //std::cout << "Finished initializations" << std::endl;
  MPI_Comm_rank(world,&me);
  char **newarg = new char*[4];
  char *strainDir;
  int numStrain = 5; /* number of strain intervals in one direction */
  int minIndex = 0;
  double strainMag;
  char * chStrainMag = new char[10];
  double dNumStrain = double(numStrain);
  double littleEps = eps / 5.0;
  newarg[0] = (char *) "0.0";
  newarg[1] = (char *) "1.0e-6";
  newarg[2] = (char *) "1000";
  newarg[3] = (char *) "10000";

  if(comm->me == 0) fprintf(screen, "CR_UPDATE- Starting Minimization of Box\n");
  if(comm->me == 0) fprintf(screen, "CR_UPDATE- %f\t%f\t%f\t%f\t%f\t%f\n", domain->h[0], domain->h[1], domain->h[2], domain->h[3], domain->h[4], domain->h[5]);
  for (int j =0; j < 6; j++)
        {
          if (j == 0) {strainDir = "x";}
          if (j == 1) {strainDir = "y";}
          if (j == 2) {strainDir = "z";}
          if (j == 3) {strainDir = "xy";}
          if (j == 4) {strainDir = "xz";}
          if (j == 5) {strainDir = "yz";}

          // Loop over strain intervals
	  while(true)
	  {
            for (int k = 0; k < (2 * numStrain + 1); k++)
              {
              // Calculate strain amount for given interval step
              // Starts from maximum negative strain, works up
                strainMag = 1.0 - littleEps + k * littleEps / dNumStrain;

                ConvertDoubleToChar(strainMag, chStrainMag);

              // Apply strain and store in strain array
                ApplyStrain(strainDir, chStrainMag);

              // Minimize and store energy
                currEnergy = CallMinimize();
	        if(currEnergy < minEnergy)
	        {
		  minEnergy = currEnergy;
		  minIndex = k;
	        }
		if(comm->me == 0) fprintf(screen, "CRD_UPDATE- %f\t%f\t%f\t%f\t%f\t%f\n", domain->h[0], domain->h[1], domain->h[2], domain->h[3], domain->h[4], domain->h[5]);

                // Undo strain
                UndoStrain(oLat);

              }

          strainMag = 1.0 - littleEps + minIndex * littleEps / dNumStrain;

          ConvertDoubleToChar(strainMag, chStrainMag);

          // Apply strain and store in strain array
          ApplyStrain(strainDir, chStrainMag);
	  CopyBoxToLat(oLat);

	  if(minIndex == 0 or minIndex == (2 * numStrain + 1))
          {
	    if(comm->me ==0) fprintf(screen, "Minimum energy occurs for the %d step, so repeating the minimization procedure.\n", minIndex); 
	    if(comm->me == 0) fprintf(screen, "CR_UPDATE- %f\t%f\t%f\t%f\t%f\t%f\n", domain->h[0], domain->h[1], domain->h[2], domain->h[3], domain->h[4], domain->h[5]);
          }
	  else break;
          }
        }

  if(comm->me == 0) fprintf(screen, "Xlo is now %f\n",  domain->boxlo[0]);

  CopyBoxToLat(oLat);

  delete [] chStrainMag;
  delete [] newarg;

  return;
}

/* ----------------------------------------------------------------------
   clear force on own & ghost atoms
   clear other arrays as needed
   ------------------------------------------------------------------------- */

void CoupleElastic::force_clear()
{
  // clear global force array
  // if either newton flag is set, also include ghosts

  size_t nbytes = sizeof(double) * atom->nlocal;
  if (force->newton) nbytes += sizeof(double) * atom->nghost;

  if (nbytes) {
    memset(&atom->f[0][0],0,3*nbytes);
  }
}

/* ----------------------------------------------------------------------
   free and reallocate relevant arrays
   ------------------------------------------------------------------------- */

void CoupleElastic::allocate(int nStrain)
{
  memory->destroy(oTLS);
  memory->create(oTLS,atom->natoms,3,"couple_elastic:oTLS");
  memory->destroy(oLat);
  memory->create(oLat,9,"couple_elast:oLat");
  memory->destroy(strainTLS);
  memory->create(strainTLS,6,2*nStrain+1,"couple_elastic:strainTLS");
  memory->destroy(energyTLS);
  memory->create(energyTLS,2,6,2*nStrain+1,"couple_elastic:energyTLS");
  memory->destroy(stressTLS);
  memory->create(stressTLS,2,6,2*nStrain+1,6,"couple_elastic:stressTLS");
  memory->destroy(deltaTLS);
  memory->create(deltaTLS,6,2*nStrain+1,"couple_elastic:deltaTLS");
  memory->destroy(cc);
  memory->create(cc,6,"couple_elastic:cc");
  memory->destroy(ec);
  memory->create(ec,2,6,6,"couple_elastic:ec");
  memory->destroy(em);
  memory->create(em,2,4,"couple_elastic:em");
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double CoupleElastic::memory_usage()
{
  double bytes = 1 * (1) * sizeof(double);
  return bytes;
}
