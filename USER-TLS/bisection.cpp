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

#include <stdlib.h>
#include <string.h>
#include <cstring>
#include <sstream>
#include <cmath>
#include "run.h"
#include "comm.h"
#include "domain.h"
#include "update.h"
#include "force.h"
#include "integrate.h"
#include "modify.h"
#include "output.h"
#include "finish.h"
#include "input.h"
#include "timer.h"
#include "error.h"
#include "bisection.h"
#include "read_dump.h"
#include "output.h"
#include "memory.h"
#include "min.h"
#include "minimize.h"
#include "atom.h"
#include "fix.h"
#include "fix_store.h"
#include "fix_store_lat.h"
#include "dump.h"
#include <iostream>
#include "write_restart.h"
#include "compute.h"
#include "compute_freq.h"

using namespace LAMMPS_NS;

#define MAXLINE 2048

/* ---------------------------------------------------------------------- */

Bisection::Bisection(LAMMPS *lmp) : Pointers(lmp)
{
        nMRelSteps = 10000;     				//The number of relaxation steps to start with for CallMinimize()
	maxAlphaSteps = 4;    				//The maximum number of addition minimizations to do because of the 'alpha linsearch' stopping condition.  For larger values, it helps the minimization actually move toward the true minimum.
        matchExit = true;                               //Determines whether or not to exit if the end-points of the trajectory relax to the same minimum
	nptSearch = false;
	int ndof = atom->natoms * domain->dimension;
	memory->create(hess,ndof,ndof+2,"bisection:freq");
}

Bisection::~Bisection()
{
	memory->destroy(hess);
}

/* ---------------------------------------------------------------------- */

void Bisection::command(int narg, char **arg)
{
	//Gets the arguments from the input line	
	if(narg<4) error->all(FLERR,"Bisection Method -- Illegal run command");
	bigint nsteps_input = force->bnumeric(FLERR,arg[0]);	//The number of steps in the input trajectory
	epsT = force->numeric(FLERR,arg[1]);			//Tolerance for the distance criteria for defining different minima

	inputSetFlag = 0;
	bool fromMD = false;
	char* bisFilename;
	int iarg=2;
	if(comm->me == 0) fprintf(screen, "Parsing bisection input\n");
	while(iarg<narg){
		if(comm->me == 0) fprintf(screen, "%s\n", arg[iarg]);
		if(strcmp(arg[iarg],"FMD") == 0){
		    if (iarg+1 > narg) error->all(FLERR,"Illegal run command");
			bisFilename = arg[iarg+1];
			fromMD = true;
			inputSetFlag = 1;
			iarg++;
		}
		if(strcmp(arg[iarg],"NPT") == 0){
			nptSearch = true;
			if(comm->me == 0) fprintf(screen, "At the beginning, nptSearch is %s", nptSearch ? "true" : "false");
		}
		iarg++;
	}

	if(inputSetFlag==0) error->all(FLERR,"Bisection Method -- No input method selected");
	BisectionFromMD(nsteps_input, bisFilename);
	
	return;

}

void Bisection::BisectionFromMD(bigint nsteps, char* bisFilename){

	//Initialize all bisection variables
	bigint intCurrStep = 0;
	bigint maxSteps = log2(nsteps)+5;
	bigint lowerStep=0, higherStep=nsteps;
	char **readInput;
	int nInput;
	int me;
	int minIndex1;
	int minIndex2;
	int hessFlag;
	int compareFlag;
	double lEnergyMin;
	double hEnergyMin;
	double tEnergyMin;
	double eDiff;
	double lDistDiff;
	double hDistDiff;
        MPI_Comm_rank(world,&me);

	//Opens TLS.dump, where the output of the TLS search is written
	if(me==0) OpenTLS();

	//Prepares input commands for the read_dump command.  If-statement handles 2d simulations
	if(domain->dimension==2)
	{
		readInput = new char*[6];
		readInput[0] = (char *) bisFilename;
		readInput[1] = (char *) "0";
		readInput[2] = (char *) "x";
		readInput[3] = (char *) "y";
		readInput[4] = (char *) "replace";
		readInput[5] = (char *) "yes";
		nInput = 6;
	}
	else
	{
                readInput = new char*[9];
                readInput[0] = (char *) bisFilename;
                readInput[1] = (char *) "0";
                readInput[2] = (char *) "x";
                readInput[3] = (char *) "y";
		readInput[4] = (char *) "z";
                readInput[5] = (char *) "replace";
                readInput[6] = (char *) "yes";
                readInput[7] = (char *) "box";
                readInput[8] = (char *) "yes";
		nInput = 9;
	}

	//Get a mapping, to handle per-atom arrays.
        if (atom->map_style == 0) {
                if(me==0) fprintf(screen, "Bisection: Getting new map\n");
                atom->nghost = 0;
                atom->map_init();
                atom->map_set();
        }

	//Initialize arrays for storing atomic positions and lattice vectors.  These are stored in FixStore and FixStoreLat objects, respectively.  This handles the migration of atoms across processors inherently
	InitAtomArrays();

	//Creates a ReadDump class, and then has it read the appropriate timestep, using the parsed readInput.
	ReadDump *bisRead = new ReadDump(lmp);
	bisRead->command(nInput, readInput);
	CopyAtoms(lAtoms,atom->x);

	//Minimizes the first atomic configuration from the dump file and then stores the energy in lEnergyMin.
	lEnergyMin = CallMinimize();
	CopyAtoms(m1Atoms,atom->x);
	CopyBoxToLat(lat1);

        //Updates readInput to go to point toward the current step, and updates the current step to be the last step in the trajectory.  
        //Then, loads and minimizes that atomic configuration from the dump file and then stores the energy in hEnergyMin.	
        readInput[1] = new char[50];
	char *charCurrStep = readInput[1];
	intCurrStep = UpdateDumpArgs(nsteps, charCurrStep);
	bisRead->command(nInput, readInput);
	CopyAtoms(hAtoms,atom->x);
	hEnergyMin = CallMinimize();
	CopyAtoms(m2Atoms,atom->x);
	CopyBoxToLat(lat2);

	//Checks the mass-weighted distance between the minimized configurations of the first and last timesteps of the trajectory.  If they are in the same minimum (ie, the mw distance is smaller than the cut-off), then it is unlikely that a TLS
	//will be found in the trajectory.  If matchExit is true, the bisection method is exited at this point.  If false, it only outputs a warning.	
	double initialDistance = ComputeDistance(m1Atoms,m2Atoms);
	if(me==0) fprintf(screen,"UPDATE-Initial Mass-Weighted Distance is %f\n", initialDistance);
	if(initialDistance<epsT/3.0)
	{
		if(matchExit) 
		{
			if(me==0) fprintf(screen,"UPDATE-Exiting bisection, as end-points for bisection have same minimum.\n");
			modify->delete_fix((char *) "TLS1");
			modify->delete_fix((char *) "TLS2");
			return;
		}
		if(me==0) fprintf(screen,"UPDATE-End-points for bisection have same minimum.  Bisection may fail.\n");
	}

	while(true)
	{
		intCurrStep = UpdateDumpArgs((higherStep-lowerStep)/2+lowerStep,charCurrStep);
		bisRead->command(nInput, readInput);
		compareFlag = ComparePositions();
		if(compareFlag == -1)
		{
			lowerStep = intCurrStep;
		}
                if(compareFlag == 1) 
                {
                        higherStep = intCurrStep;
                }
		if(compareFlag == 0)
		{
			higherStep = intCurrStep;
		}
		if(compareFlag == 2)
		{
			higherStep = intCurrStep;
		}
		if(me == 0) fprintf(screen, "UPDATE- lower=" BIGINT_FORMAT ", higher=" BIGINT_FORMAT "\n", lowerStep, higherStep);
		if(higherStep-lowerStep<=1) break;
	}

	eDiff = fabs(hEnergyMin-lEnergyMin);
	WriteTLS(intCurrStep,m1Atoms,m2Atoms,lEnergyMin,hEnergyMin);

	if(comm->me == 0)
	{
		for(int i=0; i < 9; i++) fprintf(screen, "For value %d, Lat 1 has value %f and Lat 2 has value %f\n", i, lat1[i], lat2[i]);
	}

	//Deletes readInput and atom arrays, to prevent memory leaks.
	
	delete bisRead;
	delete [] readInput;
	memory->destroy(tLat);
	if(me==0) fclose(fp);
	modify->delete_fix((char *) "TLSt");

	return;
}

int Bisection::ComparePositions()
{
	CopyAtoms(tAtoms, atom->x);
	CopyBoxToLat(tLat);
	double tEnergyMin = CallMinimize();
	double lDistDiff = ComputeDistance(m1Atoms,atom->x);
	double hDistDiff = ComputeDistance(m2Atoms,atom->x);
	if((lDistDiff<hDistDiff)&&(lDistDiff<epsT))
	{
		CopyAtoms(lAtoms, tAtoms);
		CopyLatToLat(lat1,tLat);
		CopyAtoms(m1Atoms,atom->x);
		if(comm->me == 0) fprintf(screen, "UPDATE-Match L (%f)\n", lDistDiff);
		return -1;
	}
	else{
		if(hDistDiff<epsT)
		{
			CopyAtoms(hAtoms, tAtoms);
			CopyLatToLat(lat2, tLat);
			CopyAtoms(m2Atoms,atom->x);
			if(comm->me == 0)  fprintf(screen, "UPDATE-Match U (%f)\n", hDistDiff);
			return 1;
		}
		else{
			int hessFlag = CheckHessian();
			if(hessFlag == 1)
			{
				//TO DO: Finish writing ShiftPositions(), so that the positions can be shifted using the negative eigenvector and the adjacent minima can be found.
				//ShiftPositions(1);
				if(comm->me == 0)  fprintf(screen, "UPDATE-Found saddle point early :o\n");
				return 0;
			}
			else if(hessFlag == 0)
			{
				CopyAtoms(hAtoms, tAtoms);
				CopyLatToLat(lat2, tLat);
				CopyAtoms(m2Atoms,atom->x);
				if(comm->me == 0)  fprintf(screen, "UPDATE-Match N (%f, %f); replaced upper min.\n", lDistDiff, hDistDiff);
				return 1;
			}
			else
			{
				if(comm->me == 0) fprintf(screen, "UPDATE-Weird Hessian at relaxed position, with %d negative eigenvalues.\n", hessFlag);
				return 2;
			}
		}
	}
}

//Converts a std::string into a char **, which is required for interfacing with several of the classes and functions
//within LAMMPS.
int Bisection::ConvertToChar(char ** charArray, std::string strInput)
{
	int nArgs = 0;
	char *charInput = new char[strInput.length()+1];
	std::strcpy(charInput,strInput.c_str());
	char *start = &charInput[0];
	char *stop;
	charArray[0] = start;
	while(1){
		nArgs++;
		stop = &start[strcspn(start," ")];
		if(*stop=='\0') break;
		*stop = '\0';
		start = stop+1;
		charArray[nArgs] = start;
	}

	delete [] charInput;

	return nArgs;
}

double Bisection::CallMinimize()
{       
        int Steps = nMRelSteps;
        int maxLoops = 6;
        int me;
	int alphaCounter = 0;
        char cSteps[10];
        char cFSteps[10];
        MPI_Comm_rank(world,&me);
        char **newarg = new char*[4];
        newarg[0] = (char *) "0.0";
        newarg[1] = (char *) "1.0e-6";
        ConvertIntToChar(cSteps,Steps);
        newarg[2] = cSteps;
        ConvertIntToChar(cFSteps,10*Steps);
        newarg[3] = cFSteps;
	if(comm->me == 0) fprintf(screen, "nptSearch is %s", nptSearch ? "true" : "false");
	if(nptSearch)
	{
		char **brArg = new char*[5];
		//brArg[0] = (char *) "fix";
		brArg[0] = (char *) "SearchBoxRelax";
		brArg[1] = (char *) "all";
		brArg[2] = (char *) "box/relax";
		brArg[3] = (char *) "aniso";
		brArg[4] = (char *) "0.0";
		char **frArg = new char*[6];
		//frArg[0] = (char *) "fix";
		frArg[0] = (char *) "SearchFreeze";
		frArg[1] = (char *) "all";
		frArg[2] = (char *) "setforce";
		frArg[3] = (char *) "0.0";
		frArg[4] = (char *) "0.0";
		frArg[5] = (char *) "0.0";
		for(int i = 0; i < maxLoops; i++)
		{
			Minimize* rMinAtoms = new Minimize(lmp);
			rMinAtoms->command(4, newarg);
			delete rMinAtoms;

			modify->add_fix(6, frArg, 1);
			modify->add_fix(5, brArg, 1);
                        Minimize* rMinBox = new Minimize(lmp);
                        rMinBox->command(4, newarg);
                        delete rMinBox;

			modify->delete_fix(brArg[0]);
			modify->delete_fix(frArg[0]);
			if(update->minimize->stop_condition == 3) break;
		}
		if(comm->me == 0) fprintf(screen, "Xlo is now %f\n",  domain->boxlo[0]);
		delete [] frArg;
		delete [] brArg;
	}
	else
	{
		for(int i = 0; i < maxLoops; i++)
		{       
			Minimize* rMin = new Minimize(lmp);
			rMin->command(4, newarg);
			delete rMin;
			if(update->minimize->stop_condition<2)
			{       
				if(me==0) fprintf(screen, "Minimization did not converge, increasing max steps to %d and max force iterations to %d.\n", Steps, Steps*10);
				Steps = Steps * 5;
				ConvertIntToChar(cFSteps,Steps);
				ConvertIntToChar(cFSteps,10*Steps);
			}
			else if((update->minimize->stop_condition==5)&&(alphaCounter<maxAlphaSteps))
			{
				if(me==0) fprintf(screen, "Minimization did not converge, resubmitting to handle alpha linesearch stopping condition. Counter:%d; Max:%d.\n",alphaCounter,maxAlphaSteps);
				alphaCounter++;
			}
			else break;
		}
	}

	delete [] newarg;

	return update->minimize->efinal;
}

int Bisection::UpdateDumpArgs(bigint currStep, char *charCurrStep)
{
	std::ostringstream oss;
	oss << (long long)currStep;
	std::string strCurrStep = oss.str();
	std::strcpy(charCurrStep,strCurrStep.c_str());
	return currStep;
}

//Calculates the difference between two minima.  Now, it finds the mass-weighted distance for atoms above the distCriteria.
double Bisection::ComputeDistance(double** pos1, double** pos2)
{       
        double dist = 0.0;
	double atomDist;
        double diff;
        double mTot = 0.0;
	double distCriteria = 0.00;
        double* m = atom->mass;
        int* type = atom->type;
        int me;
        MPI_Comm_rank(world,&me);

        for(int i=0; i<atom->nlocal;i++)
        {       
                diff = 0.0; 
		atomDist = 0.0;
                for(int j=0; j<domain->dimension;j++)
                {       
                        diff = pos2[i][j]-pos1[i][j];
                        if(diff < -domain->prd_half[j])
                        {       
                                diff = diff + domain->prd[j];
                        }
                        else if(diff > domain->prd_half[j])
                        {       
                                diff = diff - domain->prd[j];
                        }
			atomDist = atomDist + diff*diff;
		}
		atomDist = sqrt(atomDist);
		if(atomDist > distCriteria)
		{
			mTot = mTot + m[type[i]];
			dist = dist + m[type[i]]*atomDist;
		}
        }

	double commMassDist  [2]= {dist,mTot};
	double finMassDist [2];
	MPI_Allreduce(commMassDist,finMassDist,2,MPI_DOUBLE,MPI_SUM,world);
	if(finMassDist[1]<1e-6) return 0.0;
	return finMassDist[0]/finMassDist[1];
}

void Bisection::OpenTLS()
{
	std::string strFile = "TLS.dump";
	char *charFile = new char[20];
	std::strcpy(charFile,strFile.c_str());
	fp = fopen(charFile,"a");

	delete [] charFile;

	return;
}

void Bisection::WriteTLS(bigint step, double** x1, double** x2, double E1, double E2)
{
	double dist = ComputeDistance(x1,x2);
	double Ediff = fabs(E2 - E1);
	int me;
	MPI_Comm_rank(world, &me);
	if(me==0) fprintf(fp, "Bisection: "BIGINT_FORMAT "\t%f\t%f \n", step, Ediff, dist);
	return;
}

void Bisection::InitAtomArrays()
{
	char **newarg = new char*[5];
        newarg[0] = (char *) "TLS1";
        newarg[1] = (char *) "all";
        newarg[2] = (char *) "STORE";
        newarg[3] = (char *) "0";
        newarg[4] = (char *) "3";

//Adds the Fix, and stores the pos1 array in the astore variable of the StoreFix command.
        modify->add_fix(5,newarg);
	int iTLS1 = modify->find_fix((char *) "TLS1");
        FixStore *TLS1 = (FixStore *) modify->fix[iTLS1];
        lAtoms = TLS1->astore;

//Changes the argument of the input so that the second fix created has the label 'TLS2'.        

        newarg[0] = (char *) "TLS2";

//Adds the Fix, and stores the pos2 array in the astore variable of the StoreFix command.
        modify->add_fix(5,newarg);
	int iTLS2 = modify->find_fix((char *) "TLS2");
        FixStore *TLS2 = (FixStore *) modify->fix[iTLS2];
	hAtoms = TLS2->astore;

	newarg[0] = (char *) "TLSt";

	modify->add_fix(5,newarg);
	int iTLSt = modify->find_fix((char *) "TLSt");
	FixStore *tTLS = (FixStore *) modify->fix[iTLSt];
	tAtoms = tTLS->astore;

//Adds arrays for min1 and min2 atom positions
	newarg[0] = (char *) "TLSm1";
        modify->add_fix(5,newarg);
        int iTLSm1 = modify->find_fix((char *) "TLSm1");
        FixStore *TLSm1 = (FixStore *) modify->fix[iTLSm1];
        m1Atoms = TLSm1->astore;

        newarg[0] = (char *) "TLSm2";
        modify->add_fix(5,newarg);
        int iTLSm2 = modify->find_fix((char *) "TLSm2");
        FixStore *TLSm2 = (FixStore *) modify->fix[iTLSm2];
        m2Atoms = TLSm2->astore;

//Adds fixes for lattice storage.
	newarg[0] = (char *) "TLSLat1";
	newarg[2] = (char *) "STORELAT";
	modify->add_fix(3,newarg);
	int iTLSl1 = modify->find_fix((char *) "TLSLat1");
	FixStoreLat *TLSl1 = (FixStoreLat *) modify->fix[iTLSl1];
	lat1 = TLSl1->vstore;

        newarg[0] = (char *) "TLSLat2";
        modify->add_fix(3,newarg);
        int iTLSl2 = modify->find_fix((char *) "TLSLat2");
        FixStoreLat *TLSl2 = (FixStoreLat *) modify->fix[iTLSl2];
        lat2 = TLSl2->vstore;

        tLat = NULL;
        memory->grow(tLat,9,"ridge:tLat");
	for(int i=0;i<9;i++)
	{
		tLat[i] = 0.0;
	}

	delete [] newarg;

	return;
}

void Bisection::CopyAtoms(double** copyArray, double** templateArray)
{
	for(int i=0;i<atom->nlocal;i++)
	{
		for(int j=0;j<domain->dimension;j++)
		{
			copyArray[i][j] = templateArray[i][j];
		}
	}
	for (int i = 0; i < atom->nlocal; i++) domain->remap(atom->x[i],atom->image[i]);
	return;
}

void Bisection::ConvertIntToChar(char *copy, int n)
{       
        std::ostringstream oss;
        oss << n;
        std::string dStr = oss.str();
        std::strcpy(copy,dStr.c_str());
        return;
}

void Bisection::CopyBoxToLat(double *latVector)
{
	latVector[0] = domain->boxlo[0];
	latVector[1] = domain->boxlo[1];
	latVector[2] = domain->boxlo[2];
	latVector[3] = domain->boxhi[0];
	latVector[4] = domain->boxhi[1];
	latVector[5] = domain->boxhi[2];
	latVector[6] = domain->xy;
	latVector[7] = domain->xz;
	latVector[8] = domain->yz;
	return;
}

void Bisection::CopyLatToLat(double *copyArray, double *templateArray)
{
        for(int i=0; i<9; i++)
        {
                copyArray[i] = templateArray[i];
        }
        return;
}

/*
CheckHessian()
Uses a Hessian compute to run through the eigenfrequencies to check for a minimum.  If there are
no negative eigenfrequencies, the current configuration passes the minimum test.
*/
////////////////////////////////////////////////////////////////////////////////////////////////
int Bisection::CheckHessian()
{
        int nNeg = 0;
        int iMinimumCheck = InitHessianCompute();
        double eps = 1e-5;
        Compute* hessian = modify->compute[iMinimumCheck];
        hessian->compute_array();
        int ndof = domain->dimension * atom->natoms;
        for(int i = 0; i < ndof; i++)
        {
                if(hessian->array[i][0]<(-eps)) nNeg++;
        }
	if(nNeg != 0)
	{
		for(int i = 0; i < ndof; i++)
		{
			for(int j = 0; j < ndof; j++)
			{
				hess[i][j] = hessian->array[i][j];
			}
		}
	}
        modify->delete_compute("HessianCheck");

        return nNeg;
}

/*
InitHessianCompute()
Creates a compute of calculate the eigenfrequencies from the Hessian.  It then returns the index
for the compute.  If there is already a compute called 'HessianCheck, it returns the index
for it without creating a new one.
*/
////////////////////////////////////////////////////////////////////////////////////////////////
int Bisection::InitHessianCompute()
{
        char **newarg = new char*[5];
        newarg[0] = (char *) "HessianCheck";
        newarg[1] = (char *) "all";
        newarg[2] = (char *) "freq";
        newarg[3] = (char *) "1e-5";
        modify->add_compute(4,newarg);

        int iSaddleCheck = modify->find_compute("HessianCheck");

        delete [] newarg;
        return iSaddleCheck;
}

void Bisection::ShiftPositions(int order)
{
	int ndof = atom->natoms * domain->dimension;
	int m;
	int negIndex;
	double eps = 1e-6;
	double scale = 5.0;
	double tEnergy;
	double norm = 0.0;
	int hessFlag;
	if(order == 1)
	{
		for(int i = 0; i < ndof; i++)
		{
			if(hess[i][0] < -eps)
			{
				negIndex = i;
				break;
			}
		}
		//scale = 0.01 / sqrt(hess[negIndex][0]);
		CopyAtoms(tAtoms, atom->x);
		if(comm->me == 0) fprintf(screen, "z_4 is %f\n.", atom->x[3][2]);
		for(int i = 1; i <= atom->natoms; i++)
		{
			m = atom->map(i);
			if (m >= 0 && m < atom->nlocal)
			{
				for(int j = 0; j < domain->dimension; j++)
				{
					if(comm->me == 0) fprintf(screen, "%d, %d: %f, %f\n.", m, j, atom->x[i-1][j], hess[negIndex][domain->dimension * m + j + 2]);
					atom->x[i-1][j] = atom->x[i-1][j] + scale * hess[negIndex][domain->dimension * m + j + 2];
					norm = norm + hess[negIndex][domain->dimension * m + j + 2] * hess[negIndex][domain->dimension * m + j + 2];
				}
			}
		}
		if(comm->me == 0) fprintf(screen, "z_4 is %f, norm is %f\n.", atom->x[3][2], norm);
		CopyAtoms(hAtoms, atom->x);
		tEnergy = CallMinimize();
		//CopyAtoms(hAtoms, tAtoms);
		hessFlag = CheckHessian();
		if(comm->me == 0) fprintf(screen, "Result of CheckMin is %d\n", hessFlag);
		CopyAtoms(m2Atoms, atom->x);
		CopyBoxToLat(lat2);

		CopyAtoms(atom->x, tAtoms);
		if(comm->me == 0) fprintf(screen, "z_4 is %f\n.", atom->x[3][2]);
                for(int i = 0; i < atom->natoms; i++)
                {
                        m = atom->map(i);
                        if (m >= 0 && m < atom->nlocal)
                        {
                                for(int j = 0; j < domain->dimension; j++)
                                {
                                        atom->x[i-1][j] = atom->x[i-1][j] - scale * hess[negIndex][domain->dimension * m + j + 2];
                                }
                        }
                }
		CopyAtoms(lAtoms, atom->x);
		if(comm->me == 0) fprintf(screen, "z_4 is %f\n.", atom->x[3][2]);
                tEnergy = CallMinimize();
                hessFlag = CheckHessian();
                if(comm->me == 0) fprintf(screen, "Result of CheckMin is %d\n", hessFlag);
		//CopyAtoms(lAtoms, tAtoms);
                CopyAtoms(m1Atoms, atom->x);
                CopyBoxToLat(lat1);

	}
	/*else
	{

	}*/
	return;
}
