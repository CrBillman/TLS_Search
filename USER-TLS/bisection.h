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

/* Coded by Chris Billman, University of Florida 2016---------- */

#ifdef COMMAND_CLASS

CommandStyle(bisection,Bisection)

#else

#ifndef LMP_BISECTION_H
#define LMP_BISECTION_H

#include "pointers.h"
#include "read_dump.h"
#include <stdlib.h>
#include <string>

namespace LAMMPS_NS {

class Bisection : protected Pointers {
private:
	bool matchExit;
	bool nptSearch;
	int nAtomArrays;
	int maxAlphaSteps;
	FILE *fp;
	int inputSetFlag;
	double nMRelSteps;
	double epsT;
        double **lAtoms;
        double **hAtoms;
        double **tAtoms;
	double **m1Atoms, **m2Atoms;
	double *lat1;
	double *lat2;
	double *tLat;
	double **hess;

	int ConvertToChar(char **, std::string);
	int UpdateDumpArgs(bigint, char*);
	int CheckHessian();
	int InitHessianCompute();
	int ComparePositions();
	double CallMinimize();
	double ComputeDistance(double**,double**);
	void BisectionFromMD(bigint, char*);
	void CopyBoxToLat(double *);
	void CopyLatToLat(double *, double *);
	void WriteTLS(bigint, double**, double**, double, double);
	void OpenTLS();
	void TestMinimize(bigint, ReadDump*, int, char**);
	void TestComputeDistance();
	void StoreAtoms(double**, double**);
	void WriteAtoms(double**, double**);
        void CopyAtoms(double**, double**);
        void MappedCopyAtoms(double**, double**);
        void InitAtomArrays();
        void DeleteAtomArray(double**);
	void ConvertIntToChar(char *, int);
	void ShiftPositions(int);
public:
	Bisection(class LAMMPS *);
	~Bisection();
	void command(int, char **);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

*/
