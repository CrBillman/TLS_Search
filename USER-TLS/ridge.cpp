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
#include "comm.h"
#include "run.h"
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
#include "write_dump.h"
#include "output.h"
#include "memory.h"
#include "min.h"
#include "minimize.h"
#include "atom.h"
#include "fix.h"
#include "fix_store.h"
#include "fix_store_lat.h"
#include "ridge.h"
#include "irregular.h"
#include <iostream>
#include "compute.h"
#include "compute_freq.h"

using namespace LAMMPS_NS;

#define MAXLINE 2048

/* ---------------------------------------------------------------------- */

Ridge::Ridge(LAMMPS *lmp) : Pointers(lmp) {
	//epsF = 25e-3;
	nPRelSteps = 6;
	nMRelSteps = 100;
	maxAlphaSteps = 4;                              //The maximum number of addition minimizations to do because of the 'alpha linsearch' stopping condition.  For larger values, it helps the minimization actually move toward the true minimum.
	dmax = 0.1;
	nptSearch = false;
	nDivEnergy = 0;
}


/* ---------------------------------------------------------------------- */

void Ridge::command(int narg, char **arg){
	
	if(narg<4) error->all(FLERR,"Ridge Method -- Illegal run command");

        double time2, time1, tmp, time_loop;
	time1 = MPI_Wtime();

	nRSteps = force->numeric(FLERR,arg[0]);
	nBSteps = force->numeric(FLERR,arg[1]);
	epsT = force->numeric(FLERR,arg[2]);
	epsF = force->numeric(FLERR,arg[3]);

	int iarg = 4;
        while(iarg<narg){
                if(strcmp(arg[iarg],"NPT") ==0){
                        nptSearch = true;
                }
                iarg++;
        }

	PerformRidge();

	time2 = MPI_Wtime();
	time_loop = time2 - time1;
	int nprocs;
	MPI_Comm_size(world,&nprocs);
	MPI_Allreduce(&time_loop,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
	time_loop = tmp/nprocs;
	if(comm->me==0) fprintf(screen, "Timing - Ridge method wall time = %g.\n",time_loop);
	
	return;
}

/*
PerformRidge
(
)
Performs a modified version of the ridge method presented in Ionova J Phys. Chem (1993) and described in Hamdan, J Phys. Chem. (2014).
Using a picture of two basins separated by a ridge, the ridge method is designed to find the saddle point.  The ridge method consists of two alternating steps:

1. Bisecting atomic positions on either side of the ridge that separates the two basins.  This
causes the positions to climb to the top of the ridge.
2. Partial relaxations, of an atomic configuration on each side of the ridge.  Some of this relaxation
will pull the configuration down the ridge, but some of the relaxation will be projected along the ridge
toward the saddle point.

The saddle point is continually checked by examining the forces.  If the saddle point is found,
the details of the TLS are written to disk.  If not, the ridge method terminates after nRSteps.
*/
////////////////////////////////////////////////////////////////////////////////////////////////
void Ridge::PerformRidge()
{
	double RLt1, RLt2;
	double PRt1, PRt2;
        int me;
	bool exitFlag=false;
        MPI_Comm_rank(world,&me);
	if(me==0) OpenTLS();
	InitAtomArrays();
        if(LoadPositions()<0)
	{
		if(me==0) fprintf(screen, "Atomic positions not stored in fix_store. Bisection method needs to find minima first.\n");
		return;
	}

	CopyAtoms(atom->x,lAtoms);
	CopyLatToBox(lLat);
	eTLS1 = CallMinimize();
	CopyAtoms(pTLS1,atom->x);
	CopyBoxToLat(lTLS1);

	CopyLatToLat(tLat,lLat);
	
	CopyAtoms(atom->x,hAtoms);
	CopyLatToBox(hLat);
	eTLS2 = CallMinimize();
	CopyAtoms(pTLS2,atom->x);
	CopyBoxToLat(lTLS2);

	double chDist = ComputeDistance(pTLS2,pTLS1);
	if(chDist<epsT/3.0)
	{
		if(me==0) fprintf(screen, "UPDATE-End-points relaxed to same minimum (distance is %f), leaving ridge method.\n", chDist);
		UnfixTLS();
		return;
	}
	if(me==0) fprintf(screen, "UPDATE-Asymmetry: %f\n",fabs(eTLS2-eTLS1));

	for(int i=1; i<=nRSteps;i++)
	{
		prevMatch = -1;
		flipFlag = false;
		RLt1 = MPI_Wtime();
		divEnergy = false;
		for(int j=0; j < nBSteps; j++)
		{
			if( (j==(nBSteps - 1)) && (flipFlag) ) break;
			BisectPositions(lAtoms, hAtoms, tAtoms);
			prevForce = ComputeForce(tAtoms, tLat);
			exitFlag = CheckSaddle(tAtoms);
			if(exitFlag)
			{
				break;
			}

			ComparePositions(lAtoms, hAtoms, tAtoms);
			if(divEnergy) break;
		}
		RLt2 = MPI_Wtime();
		if(comm->me==0) fprintf(screen, "Timing - Loop over bisections wall time = %g.\n",RLt2 - RLt1);

		if(exitFlag || nDivEnergy > 5) break;
		if(i==nRSteps)
		{
			MinimizeForces(tAtoms);
			exitFlag = CheckSaddle(tAtoms);
			break;
		}
		if(me==0) fprintf(screen, "UPDATE-\t%i\tDoing partial relaxation.\n", i);
		PRt1 = MPI_Wtime();
		PartialRelax(lAtoms, hAtoms);
		PRt2 = MPI_Wtime();
		if(comm->me==0) fprintf(screen, "Timing - Partial Relaxation wall time = %g.\n",PRt2 - PRt1);
	}

	if(nDivEnergy > 5)
	{
		if(me==0) fprintf(screen, "UPDATE-Path cannot avoid points with diverging energy.\n");
	}

	if(!exitFlag)
	{
		if(me==0) fprintf(screen, "UPDATE-Cannot find saddle.\n");
		UnfixTLS();
	}

	if(atom->map_style != 0)
	{
                atom->map_delete();
                atom->map_style = 0;
	}
		
	if(me==0) fclose(fp);
	memory->destroy(tLat);
	memory->destroy(hLat);
	memory->destroy(lLat);
	for(int i = 0; i < 9; i++)
	{
		std::cout << lTLS1[i] << std::endl;
	}
	
	return;
}

/*
BisectPositions
(
pos1 : 		points to a FixStore that contains the first set of atomic positions
pos2 :		points to a FixStore that contains the second set of atomic positions
posOut : 	points to a FixStore that will contain atomic positions that are bisections of pos1 and pos2 
)
Bisects the positions in pos1 and pos2 to get the new positions that will be stored in posOut.
Also bisects the latice dimensions in lLat and hLat to get the new lattice dimensions for tLat.
*/
////////////////////////////////////////////////////////////////////////////////////////////////
void Ridge::BisectPositions(double** pos1, double** pos2, double** posOut)
{
	int m;
	double diff = 0.0;

	//Fills posOut with the values from pos1.  These should be shifted only for local atoms that move between pos1 and pos2.
	CopyAtoms(posOut,pos1);

	//Loops over atoms, using the mapping from atom->map.  If the atom is owned by the processor, its position is shifted by the difference between pos2 and pos1 for that atom.
	for(int i=0;i<atom->nlocal;i++)
	{
		for(int j=0; j<domain->dimension;j++)
		{
			diff = pos2[i][j] - pos1[i][j];
			//This if-statement ensures that the difference is not being computed across the unit cell if the atoms are just moving from one side of the unit cell to the other.
			if(diff < -domain->prd_half[j])
			{
				diff = diff + domain->prd[j];
			}
			else if(diff > domain->prd_half[j])
			{
				diff = diff - domain->prd[j];
			}
			posOut[i][j] = posOut[i][j] + 0.5*diff;
			if(diff > 1.5)
			{
				std::cout << "The atom " << i << " has distance " << diff << " A in the " << j << "th direction." << std::endl;
			}
		}
	}

	for(int i=0;i<9;i++)
	{
		tLat[i] = 0.5*(hLat[i]+lLat[i]);
	}

	//CheckDistances();

	return;
}

/*
LoadPositions()
Looks up the FixStore's and FixStoreLat's that should have been recorded by the bisection method.
If it can't find them, returns an error, the bisection method has not finished before the ridge.
Otherwise, loads:
pTLS1 : 	the atomic positions for the first minimum
pTLS2 :		the atomic positions for the second minimum
lTLS1 : 		the lattice dimensions for the first minimum
lTLS2 : 		the lattice dimensions for the second minimum
It also creates a FixStore and FixStoreLat for the saddle point:
pTLSs : 	the atomic positions for the saddle point
lTLSs : 		the lattice dimensions for the saddle point
Finally, it initializes:
lAtoms = pTLS1
lLat = lTLS1
hAtoms = pTLS2
hLat = lTLS2
*/
////////////////////////////////////////////////////////////////////////////////////////////////
int Ridge::LoadPositions()
{
	int me;
        int m;
	double diff;
        MPI_Comm_rank(world,&me);
	//First, get the labels for the fixes for the TLS atom positions.
	int iTLS1 = modify->find_fix((char *) "TLS1");
	int iTLS2 = modify->find_fix((char *) "TLS2");
	int iTLSl1 = modify->find_fix((char *) "TLSLat1");
	int iTLSl2 = modify->find_fix((char *) "TLSLat2");

	//If there are no corresponding fixes, returns -1 to flag the error.
	if((iTLS1<0)||(iTLS2<0)) return -1;

	//Creates a fix according to the stored fix
	FixStore* TLS1 = (FixStore *) modify->fix[iTLS1];
	FixStore* TLS2 = (FixStore *) modify->fix[iTLS2];
	FixStoreLat* TLSl1 = (FixStoreLat *) modify->fix[iTLSl1];
	FixStoreLat* TLSl2 = (FixStoreLat *) modify->fix[iTLSl2];

        if(me==0) std::cout << "Loading Atoms" << std::endl;

	//Copies the array in the FixStore to the arrays used within this class.
	//CopyAtoms(atom->x,TLS1->astore);
	pTLS1 = TLS1->astore;
	pTLS2 = TLS2->astore;
	lTLS1 = TLSl1->vstore;
	lTLS2 = TLSl2->vstore;

	CopyAtoms(lAtoms, pTLS1);
	CopyAtoms(hAtoms, pTLS2);
	CopyLatToLat(lLat, lTLS1);
	CopyLatToLat(hLat, lTLS2);


	//Creates FixStore for Saddle Point configuration
        char **newarg = new char*[5];

	//Created the arguments for the StoreFix
        newarg[0] = (char *) "TLSs";
        newarg[1] = (char *) "all";
        newarg[2] = (char *) "STORE";
        newarg[3] = (char *) "0";
        newarg[4] = (char *) "3";

	//Adds the Fix, and stores the pos1 array in the astore variable of the StoreFix command.
        modify->add_fix(5,newarg);
	int iTLSs = modify->find_fix((char *) "TLSs");
        FixStore *TLSs = (FixStore *) modify->fix[iTLSs];
        CopyAtoms(TLSs->astore,pTLS1);
	pTLSs = TLSs->astore;

        newarg[0] = (char *) "TLSLatS";
        newarg[2] = (char *) "STORELAT";
        modify->add_fix(3,newarg);
        int iTLSlS = modify->find_fix((char *) "TLSLatS");
        FixStoreLat *TLSlS = (FixStoreLat *) modify->fix[iTLSlS];
        lTLSs = TLSlS->vstore;

	MPI_Barrier(world);

	delete [] newarg;
	return 0;
}

/*
CopyAtoms
(
copyArray : 		points to a FixStore that contains atomic positions, to be updated
templateArray : 	points to a FixStore that contains atomic positions, to be copied
)
Copies the atomic positions from templateArray to copyArray.  Built to work across parallelization,
so only nlocal entries are updated in copyArray.
*/
////////////////////////////////////////////////////////////////////////////////////////////////
void Ridge::CopyAtoms(double** copyArray, double** templateArray)
{
        int me;
        int m;
	MPI_Comm_rank(world,&me);
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

/*
OpenTLS()
Opens the TLS.dump file, where the details of the discovered TLS will be written, and stores 
it in fp.
*/
////////////////////////////////////////////////////////////////////////////////////////////////
void Ridge::OpenTLS()
{
        std::string strFile = "TLS.dump";
        char *charFile = new char[20];
        std::strcpy(charFile,strFile.c_str());
        fp = fopen(charFile,"a");

	delete [] charFile;
        return;
}

/*
WriteTLS
(
E1 :            Contains the energy of the first minimum
E2 :            Contains the energy of the second minimum
E3 :		Contains the energy of the saddle point
)
This function writes out the results of the TLS search.  First, it writes the

Asymmetry	Barrier Height	Barrier 1	Barrier 2	Relaxation Time

to TLS.dump.  Then, it writes out the atomic positions and lattice dimensions for the 
first minimum (timestep 0), the second minimum (timestep 1), and the saddle point (timestep 2) 
to TLS.pos.
*/
////////////////////////////////////////////////////////////////////////////////////////////////
void Ridge::WriteTLS(double E1, double E2, double E3)
{
        double Asym = fabs(E2 - E1);
	double Barrier= 0.5*((E3-E1)+(E3-E2));
	double relTime[2];
	double dist = 0.0;
	int me;
	MPI_Comm_rank(world,&me);

	ComputeRelaxationTime(relTime);

	dist = ComputeDistance(pTLS1,pTLS2);
        if(me==0) fprintf(fp, "%f\t%f\t%f\t%f\t%f\t%f\t%f\n", Asym, Barrier, dist, E3 - E1, E3 - E2, relTime[0], relTime[1]);

	char** dumparg = new char*[8];
        dumparg[0] = (char *) "all";
        dumparg[1] = (char *) "atom";
	dumparg[2] = (char *) "TLS.pos";
	dumparg[3] = (char *) "modify";
	dumparg[4] = (char *) "append";
	dumparg[5] = (char *) "yes";
	dumparg[6] = (char *) "scale";
	dumparg[7] = (char *) "no";

	WriteDump* pDump = new WriteDump(lmp);
	update->reset_timestep(0);
	CopyAtoms(atom->x,pTLS1);
	CopyLatToBox(lTLS1);
	UpdateMapping();
	pDump->command(8,dumparg);
	update->reset_timestep(1);
	CopyAtoms(atom->x,pTLS2);
	CopyLatToBox(lTLS2);
	UpdateMapping();
	pDump->command(8,dumparg);
	update->reset_timestep(2);
        CopyAtoms(atom->x,pTLSs);
	CopyLatToBox(lTLSs);
        pDump->command(8,dumparg);
	
	delete [] dumparg;
	delete pDump;

	MPI_Barrier(world);
        return;
}

/*
PartialRelax
(
lAtoms :          Points to a FixStore that contains atomic positions for the configuration below the ridge
hAtoms :          Points to a FixStore that contains atomic positions for the configuration above the ridge
)
This function partially relaxes the atomic configurations determined to be above and below
the ridge.  The number of steps used in the relaxation is determined by nPRelSteps.
There is extra functionality contained here to try to improve the ridge method.  It's important
that the relaxation is not too big, or it will move the configurations too far from the saddle point.
For this reason, the partial relaxation is set to use steepest descent, and tries to decrease
dmax based on how small the force is (Because small steps are desired if close to the saddle point).
*/
////////////////////////////////////////////////////////////////////////////////////////////////
void Ridge::PartialRelax(double** lAtoms, double** hAtoms)
{       
        char** newarg = new char*[4];
	int me;
	int m;
	double** atomPtr =atom->x;
	char *oldMinStyle;
	char **newAlpha;
	MPI_Comm_rank(world,&me);

        newarg[0] = (char *) "0.0";
        newarg[1] = (char *) "1.0e-6";
        newarg[2] = new char[4];
	char* cRelSteps = newarg[2];
        newarg[3] = (char *) "1000";

	ConvertIntToChar(cRelSteps,nPRelSteps);	

	char **brArg;
	char **frArg;

        int n = strlen(update->minimize_style) + 1;
        oldMinStyle = new char[n];
        strcpy(oldMinStyle, update->minimize_style);

	if(prevForce < dmax)
	{
		newAlpha = new char*[2];
		newAlpha[0] = (char *) "dmax";
		newAlpha[1] = new char[15];
		ConvertDoubleToChar(newAlpha[1], prevForce);
		update->minimize->modify_params(2,newAlpha);
	}

	char **styleArg = new char*[1];
	styleArg[0] = (char *) "sd";
	update->create_minimize(1, styleArg);

        if(nptSearch)
        {
                brArg = new char*[7];
                brArg[0] = (char *) "SearchBoxRelax";
                brArg[1] = (char *) "all";
                brArg[2] = (char *) "box/relax";
                brArg[3] = (char *) "aniso";
                brArg[4] = (char *) "0.0";
                brArg[5] = (char *) "vmax";
                brArg[6] = (char *) "0.01";

                frArg = new char*[6];
                frArg[0] = (char *) "SearchFreeze";
                frArg[1] = (char *) "all";
                frArg[2] = (char *) "setforce";
                frArg[3] = (char *) "0.0";
                frArg[4] = (char *) "0.0";
                frArg[5] = (char *) "0.0";

		modify->add_fix(7, brArg, 1);
	}


	Minimize* rMin = new Minimize(lmp);
	CopyLatToBox(lLat);
	CopyAtoms(atomPtr,lAtoms);
        rMin->command(4, newarg);
	delete rMin;
	CopyAtoms(lAtoms,atomPtr);
	CopyBoxToLat(lLat);

	rMin = new Minimize(lmp);
	CopyLatToBox(hLat);
        CopyAtoms(atomPtr,hAtoms);
        rMin->command(4, newarg);
	delete rMin;
        CopyAtoms(hAtoms,atomPtr);
	CopyBoxToLat(hLat);

        styleArg[0] = oldMinStyle;
        update->create_minimize(1, styleArg);

	if(prevForce < dmax)
	{
		ConvertDoubleToChar(newAlpha[1], dmax);
		update->minimize->modify_params(2,newAlpha);
		delete [] newAlpha[1];
		delete [] newAlpha;
	}

	delete [] cRelSteps;
	delete [] newarg;
	delete [] styleArg;
	delete [] oldMinStyle;
	if(nptSearch)
	{
		modify->delete_fix(brArg[0]);
		delete [] brArg;
		delete [] frArg;
	}

        return;
}

/*
ComparePositions
(
lAtoms :          Points to a FixStore that contains atomic positions for the configuration below the ridge
hAtoms :          Points to a FixStore that contains atomic positions for the configuration above the ridge
tAtoms : 	  Points to a FixStore that contains atomic positions for the working config
)
The PEL is thought to be composed of many different basins.  When searching for a TLS, we need to
be able to tell if a configuration is within the same basin as different minimized configuration.
This function takes in the working configuration, and compares it to two minima.  If at least one
minimum has a distance below some criteria (epsT), the working configuration is determined to be
in the same basin as the minimum with the smallest computed distance.
After this determination has been made, the working configuration is determined to be either above
or below the ridge.  Then, either lAtoms or hAtoms is updated with the values in tAtoms according
to where the working configuration is determined to be.
If neither minimum has a distance below the criteria, the working configuration is found to be
in a new basin.  In this case, the hessian is computed to determine if the relaxation of the
working configuration is in a new minimum.  If it is determined to be in a minmum, the second
minimum for the TLS is replaced with the new minimum, and hAtoms is replaced by tAtoms.
*/
////////////////////////////////////////////////////////////////////////////////////////////////
void Ridge::ComparePositions(double** lAtoms, double** hAtoms, double** tAtoms)
{
	double lDistDiff, hDistDiff, mDistDiff;
	double tEnergy;
        int me;
	int currMatch = -1;
        MPI_Comm_rank(world,&me);

	CopyAtoms(atom->x,tAtoms);
	CopyLatToBox(tLat);
	UpdateMapping();
	CallMinimize();
	tEnergy = update->minimize->einitial;
	if(tEnergy < -1e20 || tEnergy > 1e20)
	{
		if(me==0)  fprintf(screen, "UPDATE-Diverging Energy(%e), Rejecting Configuration.\n", tEnergy);
		nDivEnergy++;
		divEnergy = true;
		CopyAtoms(tAtoms, hAtoms);
		return;
	}

	lDistDiff = ComputeDistance(atom->x, pTLS1);
	hDistDiff = ComputeDistance(atom->x, pTLS2);
	mDistDiff = ComputeDistance(pTLS1, pTLS2);
	double latDist = 0.0;
        if(comm->me == 0)
        {
                for(int i=0; i < 9; i++) latDist = latDist + (lLat[i] - hLat[i]) * (lLat[i] - hLat[i]);
        }
	if((lDistDiff<epsT) && (lDistDiff<hDistDiff))
	{
		CopyAtoms(lAtoms,tAtoms);
		CopyLatToLat(lLat, tLat);
		currMatch = 1;
		if(me==0)  fprintf(screen, "UPDATE-Match L (%f, %f, %f): V1 = %f, V2 = %f, force = %f, RLat1 = %f, RLat2 = %f, latDist = %f \n", 
			lDistDiff, hDistDiff, mDistDiff, tEnergy - eTLS1, tEnergy - eTLS2, prevForce, hLat[3], lLat[3], latDist);
	}
	else if(hDistDiff<epsT)
	{
		CopyAtoms(hAtoms,tAtoms);
		CopyLatToLat(hLat, tLat);
		currMatch = 2;
		if(me==0)  fprintf(screen, "UPDATE-Match U (%f, %f, %f): V1 = %f, V2 = %f, force = %f, RLat1 = %f, RLat2 = %f, latDist = %f \n", 
			lDistDiff, hDistDiff, mDistDiff, tEnergy - eTLS1, tEnergy - eTLS2, prevForce, hLat[3], lLat[3], latDist);
	}
	else
	{
		CopyAtoms(hAtoms,tAtoms);
		CopyLatToLat(hLat, tLat);
		if(CheckMinimum())
		{
			CopyAtoms(pTLS2, atom->x);
			CopyBoxToLat(lTLS2);
			eTLS2 = tEnergy;
			if(me==0)  fprintf(screen, "UPDATE-Match N (%f, %f, %f): V1 = %f, V2 = %f, force = %f, RLat1 = %f, RLat2 = %f, latDist = %f.  Replaced Min 2. \n", 
				lDistDiff, hDistDiff, mDistDiff, tEnergy - eTLS1, tEnergy - eTLS2, prevForce, hLat[3], lLat[3], latDist);
		}
		else
		{
			if(me==0)  fprintf(screen, "UPDATE-Match N (%f, %f, %f): V1 = %f, V2 = %f, force = %f, RLat1 = %f, RLat2 = %f, latDist = %f.  Did not replace Min 2. \n", 
				lDistDiff, hDistDiff, mDistDiff, tEnergy - eTLS1, tEnergy - eTLS2, prevForce, hLat[3], lLat[3], latDist);
		}
	}

	if(abs(prevMatch - currMatch)==1) flipFlag = true;
	prevMatch = currMatch;

	return;
}

/*
ConvertDoubleToChar
(
pos1 :          Points to a FixStore that contains atomic positions
pos2 :          Points to a FixStore that contains atomic positions
)
Calculates the difference between two atomic configurations using the mass-weighted distance.
*/
////////////////////////////////////////////////////////////////////////////////////////////////
double Ridge::ComputeDistance(double** pos1, double** pos2)
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

double Ridge::ComputeDistance(double** pos1, double** pos2, double** l1, double** l2)
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

        double commMassDist  [2]= {dist,mTot};
        double finMassDist [2];
        MPI_Allreduce(commMassDist,finMassDist,2,MPI_DOUBLE,MPI_SUM,world);
        if(finMassDist[1]<1e-6) return 0.0;
        return finMassDist[0]/finMassDist[1];
}

/*
CallMinimize()
Interfaces with the minimize command.  This function has extra tools to handle some complications
with relaxing an amorphous system.  Commonly, the relaxation will terminate with a problem with
the alpha linesearch.  This prevents the minimization from truly finding the minimum.  However,
this problem is stochastic.  So, resubmitting the job will help the minimizer find the true minimum.
This function also can handle running out of minimization steps or force iterations by increasing
the maximum number of steps in these cases.
*/
////////////////////////////////////////////////////////////////////////////////////////////////
double Ridge::CallMinimize()
{
	int Steps = nMRelSteps;
	int maxLoops = 5;
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
        if(nptSearch)
	//if(0)
        {
                char **brArg = new char*[7];
                brArg[0] = (char *) "SearchBoxRelax";
                brArg[1] = (char *) "all";
                brArg[2] = (char *) "box/relax";
                brArg[3] = (char *) "aniso";
                brArg[4] = (char *) "0.0";
                brArg[5] = (char *) "vmax";
                brArg[6] = (char *) "0.01";

                char **frArg = new char*[6];
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
                        if(update->minimize->stop_condition == 3) break;
                }
		if(comm->me == 0) fprintf(screen, "BR_UPDATE- Starting Relaxation of Box\n");
		if(comm->me == 0) fprintf(screen, "BR_UPDATE- %f\t%f\t%f\t%f\t%f\t%f\n", domain->h[0], domain->h[1], domain->h[2], domain->h[3], domain->h[4], domain->h[5]);
                for(int i = 0; i < maxLoops; i++)
                {
			if(comm->me == 0) fprintf(screen, "BR_UPDATE- BR %d\n", i);
			if(comm->me == 0) fprintf(screen, "BR_UPDATE- %f\t%f\t%f\t%f\t%f\t%f\n", domain->h[0], domain->h[1], domain->h[2], domain->h[3], domain->h[4], domain->h[5]);
                        modify->add_fix(6, frArg, 1);
                        modify->add_fix(7, brArg, 1);
                        Minimize* rMinBox = new Minimize(lmp);
                        rMinBox->command(4, newarg);
                        delete rMinBox;

                        modify->delete_fix(brArg[0]);
                        modify->delete_fix(frArg[0]);
                        if(comm->me == 0) fprintf(screen, "BR_UPDATE- AR %d\n", i);
			if(comm->me == 0) fprintf(screen, "BR_UPDATE- %f\t%f\t%f\t%f\t%f\t%f\n", domain->h[0], domain->h[1], domain->h[2], domain->h[3], domain->h[4], domain->h[5]);
                        Minimize* rMinAtoms = new Minimize(lmp);
                        rMinAtoms->command(4, newarg);
                        delete rMinAtoms;
                        if(update->minimize->stop_condition == 3 || update->minimize->efinal < -1e20 || update->minimize->efinal > 1e20) break;
                }
                if(comm->me == 0) fprintf(screen, "Xlo is now %f\n",  domain->boxlo[0]);
		delete [] brArg;
		delete [] frArg;
        }
        else
        {
                for(int i = 0; i < maxLoops; i++)
                {
                        Minimize* rMin = new Minimize(lmp);
                        rMin->command(4, newarg);
                        delete rMin;
			if(update->minimize->efinal < -1e20 || update->minimize->efinal > 1e20)
			{
				if(me==0) fprintf(screen, "Relaxation results in unphysical energy.  Reinitializing upper configuration.\n");
				double tempE = update->minimize->efinal;
				CopyAtoms(atom->x, hAtoms);
				newarg[2] = (char *) "0";
				newarg[3] = (char *) "0";
				Minimize* rMin = new Minimize(lmp);
				rMin->command(4, newarg);
				delete rMin;
				delete [] newarg;
				return tempE;
				break;
			}
                        else if(update->minimize->stop_condition<2)
                        {
                                if(me==0) fprintf(screen, "Minimization did not converge, increasing max steps to %d and max force iterations to %d.\n", Steps, Steps*10);
                                Steps = Steps * 10;
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

/*
ConvertIntegerToChar
(
copy :          The char array that will be filled with the double
n :             The integer to be copied into the char array
)
This copies an integer into a char array using a string stream.
*/
////////////////////////////////////////////////////////////////////////////////////////////////
void Ridge::ConvertIntToChar(char *copy, int n)
{
        std::ostringstream oss;
        oss << n;
        std::string dStr = oss.str();
        std::strcpy(copy,dStr.c_str());
	return;
}

/*
ConvertDoubleToChar
(
copy : 		The char array that will be filled with the double
n : 		The double to be copied into the char array
)
This copies a double into a char array using a string stream.
*/
////////////////////////////////////////////////////////////////////////////////////////////////
void Ridge::ConvertDoubleToChar(char *copy, double n)
{
        std::ostringstream oss;
        oss << n;
        std::string dStr = oss.str();
        std::strcpy(copy,dStr.c_str());
        return;
}

/*
CheckSaddle
(
pos : 		points to FixStore that contains the atomic configuration to be checked
)
This determines whether the input atomic configuration is at a saddle point.  First, the forces
at the configuration are checked.  If they are below the force criteria, the eigenfrequencies
are computed for the position.  If only one eigenfrequency is negative, the configuration passes
the saddle test.
*/
////////////////////////////////////////////////////////////////////////////////////////////////
bool Ridge::CheckSaddle(double** pos)
{
	int me;
	double eps = 1e-5;
	MPI_Comm_rank(world,&me);
	if(me==0) fprintf(screen, "Checking Saddle.\n");
        char **newarg = new char*[4];
        newarg[0] = (char *) "0.0";
        newarg[1] = (char *) "0.0";
        newarg[2] = (char *) "0";
        newarg[3] = (char *) "0";
        Minimize* rMin = new Minimize(lmp);
	CopyAtoms(atom->x,pos);
        rMin->command(4, newarg);
        delete rMin;
	prevForce = update->minimize->fnorminf_final;
	eTLSs = update->minimize->einitial;

	if(prevForce < epsF)
	{
		int nNeg = 0;
		int nPos = 0;
		int iSaddleCheck = InitHessianCompute();
		Compute* hessian = modify->compute[iSaddleCheck];
		hessian->compute_array();
		int ndof = 3*atom->natoms;
		int negFreqIndex = 0;
		for(int i = 0; i < ndof; i++)
		{
			if(hessian->array[i][0]>eps) nPos++;
			else if(hessian->array[i][0]<(-eps))
			{
				nNeg++;
				negFreqIndex = i;
			}
		}
		if(nNeg == 1)
		{
			if(((update->minimize->einitial-eTLS1)>0.0)&&((update->minimize->einitial-eTLS2)>0.0))
			{
				if(me==0) fprintf(screen, "UPDATE-Passes Saddle Point check.\n");
				CopyAtoms(pTLSs, pos);
				FindMins(hessian->array[negFreqIndex]);
				delete [] newarg;
				modify->delete_compute("HessianCheck");
				WriteTLS(eTLS1,eTLS2,eTLSs);
				return true;
			}
			else
			{
				if(me==0) fprintf(screen, "UPDATE-Passes Saddle Point check, but barrier height from at least one side is negative.\n");
				if(FindMins(hessian->array[negFreqIndex]))
				{
					if(me==0) fprintf(screen, "UPDATE-Found nearby minima.\n");
					CopyAtoms(pTLSs, pos);
					delete [] newarg;
					modify->delete_compute("HessianCheck");
					WriteTLS(eTLS1,eTLS2,eTLSs);
					return true;
				}
				else
				{
					if(me==0) fprintf(screen, "UPDATE-Couldn't find nearby minima.\n");
					modify->delete_compute("HessianCheck");
					UnfixTLS();
					delete [] newarg;
					return true;
				}
			}
		}
		else
		{
			if(me==0) fprintf(screen, "UPDATE-No saddle point in vicinity as there are %d negative entries.\n", nNeg);
			modify->delete_compute("HessianCheck");
			UnfixTLS();
			delete [] newarg;
			return true;
		}
	}
	delete [] newarg;
	return false;
}

/*
CheckMinimum()
Uses a Hessian compute to run through the eigenfrequencies to check for a minimum.  If there are
no negative eigenfrequencies, the current configuration passes the minimum test.
*/
////////////////////////////////////////////////////////////////////////////////////////////////
bool Ridge::CheckMinimum()
{
	int nNeg = 0;
	int iMinimumCheck = InitHessianCompute((char *) "MinCheck");
	double eps = 1e-5;
	Compute* hessian = modify->compute[iMinimumCheck];
	hessian->compute_array();
	int ndof = 3*atom->natoms;
	for(int i = 0; i < ndof; i++)
	{
		if(hessian->array[i][0]<(-eps)) nNeg++;
	}
	modify->delete_compute("MinCheck");
	if(nNeg == 0)
	{
		return true;
	}
	return false;
}

/*
InitHessianCompute()
Creates a compute of calculate the eigenfrequencies from the Hessian.  It then returns the index
for the compute.  If there is already a compute called 'HessianCheck, it returns the index
for it without creating a new one.
*/
////////////////////////////////////////////////////////////////////////////////////////////////
int Ridge::InitHessianCompute()
{ 
	char **newarg = new char*[5];
	newarg[0] = (char *) "HessianCheck";
	newarg[1] = (char *) "all";
	newarg[2] = (char *) "freq";
	newarg[3] = (char *) "1e-4";
	modify->add_compute(4,newarg);

	int iSaddleCheck = modify->find_compute("HessianCheck");

	delete [] newarg;
	return iSaddleCheck;
}

/*
InitHessianCompute(char *displacement)
Creates a compute of calculate the eigenfrequencies from the Hessian.  It then returns the index
for the compute.  If there is already a compute called 'HessianCheck, it returns the index
for it without creating a new one.
*/
////////////////////////////////////////////////////////////////////////////////////////////////
int Ridge::InitHessianCompute(double displacement)
{
        char **newarg = new char*[5];
        newarg[0] = (char *) "HessianCheck";
        newarg[1] = (char *) "all";
        newarg[2] = (char *) "freq";
	char cDisplacement[16];
	ConvertDoubleToChar(cDisplacement, displacement);
        newarg[3] = cDisplacement;
        modify->add_compute(4,newarg);

        int iSaddleCheck = modify->find_compute("HessianCheck");

        delete [] newarg;
        return iSaddleCheck;
}

/*
InitHessianCompute(char *name)
Creates a compute of calculate the eigenfrequencies from the Hessian.  It then returns the index
for the compute.  If there is already a compute called 'HessianCheck, it returns the index
for it without creating a new one.
*/
////////////////////////////////////////////////////////////////////////////////////////////////
int Ridge::InitHessianCompute(char *name)
{
        char **newarg = new char*[5];
        newarg[0] = name;
        newarg[1] = (char *) "all";
        newarg[2] = (char *) "freq";
        newarg[3] = (char *) "1e-5";
        modify->add_compute(4,newarg);

        int iSaddleCheck = modify->find_compute(name);

        delete [] newarg;
        return iSaddleCheck;
}

/*
InitAtomArrays
(
)
Creates several objects that are used throughout the ridge method.  After this method is finished,
the following objects will be created:
lAtoms :	        Points to a FixStore that contains atomic positions for the configuration below the ridge
hAtoms :        	Points to a FixStore that contains atomic positions for the configuration above the ridge
tAtoms :		points to FixStore that contains the atomic positions of the working config
lLat : 			array that contains the simulation box dimensions for the first minimum
hLat : 			array that contains the simulation box dimensions for the second minimum
tLat : 			array that contains the simulation box dimensions for the working config
*/
//////////////////////////////////////////////////////////////////////////////////////////////////
void Ridge::InitAtomArrays()
{
        char **newarg = new char*[5];
        newarg[0] = (char *) "TLSl";
        newarg[1] = (char *) "all";
        newarg[2] = (char *) "STORE";
        newarg[3] = (char *) "0";
        newarg[4] = (char *) "3";

//Adds the Fix, and stores the pos1 array in the astore variable of the StoreFix command.
        modify->add_fix(5,newarg);
	int iTLSl = modify->find_fix((char *) "TLSl");
        FixStore *TLSl = (FixStore *) modify->fix[iTLSl];
        lAtoms = TLSl->astore;
        

//Changes the argument of the input so that the second fix created has the label 'TLS2'.        
        
        newarg[0] = (char *) "TLSh";

//Adds the Fix, and stores the pos2 array in the astore variable of the StoreFix command.
        modify->add_fix(5,newarg); 
	int iTLSh = modify->find_fix((char *) "TLSh");
        FixStore *TLSh = (FixStore *) modify->fix[iTLSh];
        hAtoms = TLSh->astore;
        
        newarg[0] = (char *) "TLSt";
        
        modify->add_fix(5,newarg);
	int iTLSt = modify->find_fix((char *) "TLSt");
        FixStore *TLSt = (FixStore *) modify->fix[iTLSt];
        tAtoms = TLSt->astore;

	//These lattice arrays are only used locally and don't need to be communicated across processors, so it isn't necessary to store them in a FixStoreLat object.  
	hLat = lLat = tLat = NULL;
	memory->grow(hLat,9,"ridge:hLat");
	memory->grow(lLat,9,"ridge:lLat");
	memory->grow(tLat,9,"ridge:tLat");
	//Initialize the values in the array to 0.0
	for(int i=0;i<9;i++)
	{
		tLat[i] = 0.0;
	}

	delete [] newarg;	
        return;
}

/*
UpdateMapping()
Ensures that atoms remain within the boundaries of the simulation cell.  I think this is not
necessary, but better safe than sorry.
*/
////////////////////////////////////////////////////////////////////////////////////////////////
void Ridge::UpdateMapping()
{

        for (int i = 0; i < atom->nlocal; i++) domain->remap(atom->x[i],atom->image[i]);
        if (domain->triclinic) domain->x2lamda(atom->nlocal);
        domain->reset_box();
        Irregular *irregular = new Irregular(lmp);
        irregular->migrate_atoms(1);
        delete irregular;
        if (domain->triclinic) domain->lamda2x(atom->nlocal);

	return;
}

/*
CopyBoxToLat
(
latVector:      array[9], stores xlo xhi ylo yhi zlo zhi xy xz yz. to be updated
)
Copies the simulation box to the latVector.
*/
////////////////////////////////////////////////////////////////////////////////////////////////
void Ridge::CopyBoxToLat(double *latVector)
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

/*
CopyLatToBox
(
latVector:      array[9], stores xlo xhi ylo yhi zlo zhi xy xz yz. to be copied
)
Copies the values of an input lattice array to the simulation box.  Then updates the global
parameters for the system and ensures that the mapping is updated..
*/
////////////////////////////////////////////////////////////////////////////////////////////////
void Ridge::CopyLatToBox(double *latVector)
{
        domain->boxlo[0] = latVector[0];
	domain->boxlo[1] = latVector[1];
        domain->boxlo[2] = latVector[2];
        domain->boxhi[0] = latVector[3];
	domain->boxhi[1] = latVector[4];
	domain->boxhi[2] = latVector[5];
	domain->xy = latVector[6];
	domain->xz = latVector[7];
	domain->yz = latVector[8];

        /*domain->boxlo_bound[0] = latVector[0];
        domain->boxlo_bound[1] = latVector[1];
        domain->boxlo_bound[2] = latVector[2];
        domain->boxhi_bound[0] = latVector[3];
        domain->boxhi_bound[1] = latVector[4];
        domain->boxhi_bound[2] = latVector[5];
        domain->xy = latVector[6];
        domain->xz = latVector[7];
        domain->yz = latVector[8];*/

	ResetBox();
	UpdateMapping();

        return;
}

/*
CopyLatToLat
(
copyArray:	array[9], stores xlo xhi ylo yhi zlo zhi xy xz yz. to be copied
templateArray:	array[9], stores xlo xhi ylo yhi zlo zhi xy xz yz. to be updated
)
Copies the values of an input lattice array to another one.
*/
////////////////////////////////////////////////////////////////////////////////////////////////
void Ridge::CopyLatToLat(double *copyArray, double *templateArray)
{
	for(int i=0; i<9; i++)
	{
		copyArray[i] = templateArray[i];
	}
	return;
}

/*
ResetBox()
Resetting the box ensures that all parameters are updated after changing the lattice.
*/
////////////////////////////////////////////////////////////////////////////////////////////////
void Ridge::ResetBox()
{
	domain->set_initial_box();
	domain->set_global_box();
	domain->set_local_box();
	domain->reset_box();
}

/*
MinimizeForces
(
pos:	a pointer to a FixStore object, which stores an atom array that is automatically updated
	with respect to the parallelization across processors.
)
This method adjusts the atomic configuration in order to more precisely locate the saddle point.
The ridge method often struggles to get the force below a reasonable threshold, so using another
method when the forces are already small can speed up the TLS search.
Currently, this isn't working.  The goal is the finish implementing the auxiliary potential
(Where the objective function is changed to 0.5*|F(x)|^2), and use this here.
*/
////////////////////////////////////////////////////////////////////////////////////////////////
void Ridge::MinimizeForces(double **pos)
{
	return;
}

double Ridge::ComputeForce(double **pos, double *lat)
{
        char **newarg = new char*[4];
        newarg[0] = (char *) "0.0";
        newarg[1] = (char *) "0.0";
        newarg[2] = (char *) "0";
        newarg[3] = (char *) "0";
        Minimize* rMin = new Minimize(lmp);
	CopyLatToBox(lat);
        CopyAtoms(atom->x,pos);
        rMin->command(4, newarg);
        delete rMin;
	delete [] newarg;
        return update->minimize->fnorminf_final;
}

/*
Compute relaxation time, according to Hamdan J Phys Chem (2014).  Calculated by

\tau_0^{-1}=\frac{\prod^{3N}_{i=1}v_i^0}{\prod^{3N-1}_{i=1}v_i^s}e^{\frac{S}{k_b}}

This method initializes a hessian compute.  Then it loads the the first minimum and 
lattice, and calculates the hessian.  Then, it takes the product of all of the eigenfrequencies.  
These are assumed to already follow the rule that all are positive except for 3 which correspond
to the symmetries of the system (so, if the mimimum or saddle points are not true extrema,
this method will not work).  This process is repeated for the saddle point.  Then, the ratio is 
taken and returned.
*/
////////////////////////////////////////////////////////////////////////////////////////////////
void Ridge::ComputeRelaxationTime(double* relTime)
{
        double eps = 1e-6;

        int iSaddleCheck = InitHessianCompute(1e-5);
        Compute *hessian = modify->compute[iSaddleCheck];
	hessian->compute_array();
        //This needs to be done logarithmically, so that the product will be the sum of the logarithms.  At the end, we exponentiate to get the relaxation time.
        double saddleProduct = 0.0;
        int dof = domain->dimension * atom->natoms;
        for(int i = 0; i < dof; i++)
        {
                if(hessian->array[i][0]>eps) saddleProduct = saddleProduct + log(hessian->array[i][0]);
        }
        modify->delete_compute("HessianCheck");

        ComputeForce(pTLS1, lTLS1);

        iSaddleCheck = InitHessianCompute();
        hessian = modify->compute[iSaddleCheck];
        hessian->compute_array();
        double minProduct1 = 0.0;
        for(int i = 0; i < dof; i++)
        {
                if(hessian->array[i][0]>eps) minProduct1 = minProduct1 + log(hessian->array[i][0]);
        }

        modify->delete_compute("HessianCheck");

        ComputeForce(pTLS2, lTLS2);

        iSaddleCheck = InitHessianCompute(1e-5);
        hessian = modify->compute[iSaddleCheck];
        hessian->compute_array();
        double minProduct2 = 0.0;
        for(int i = 0; i < dof; i++)
        {
                if(hessian->array[i][0]>eps) minProduct2 = minProduct2 + log(hessian->array[i][0]);
        }


	//Divide by 2 (because the eigenvalues are the frequency squared) and exponentiate to get the relaxation time.
        relTime[0] = exp(-(minProduct1 - saddleProduct)/2.0);
	relTime[1] = exp(-(minProduct2 - saddleProduct)/2.0);

        modify->delete_compute("HessianCheck");
        return;
}

void Ridge::UnfixTLS()
{
	modify->delete_fix((char *) "TLS1");
	modify->delete_fix((char *) "TLS2");
	modify->delete_fix((char *) "TLSs");
	return;
}

void Ridge::CheckDistances()
{
	double distCheck = 1.6;
	double diff = 0.0;
	double diffSq = 0.0;
	double** x = atom->x;
	distCheck = distCheck * distCheck;
        for(int i=0; i<atom->nlocal;i++)
        {
                for(int j=0; j<i;j++)
                {
			for(int k = 0; k < domain->dimension; k++)
			{
				diff = x[i][k] - x[j][k];
				if(diff < - domain->prd_half[k])
				{
					diff = diff + domain->prd[k];
				}
				else if(diff > domain->prd_half[k])
				{
					diff = diff - domain->prd[k];
				}
				diffSq = diffSq + diff * diff;
			}
			if(diffSq <= distCheck)
			{
				std::cout << "DISTANCE CHECK; atoms " << atom->tag[i] << " and " << atom->tag[j] << ", types " << atom->type[i] << ", " << atom->type[j] << " have a distance of only " << sqrt(diffSq) << "." << std::endl;
			}
                }
        }	
	return;
}

bool Ridge::FindMins(double *freqs)
{
	std::cout << "we in here" << std::endl;
	if(comm->me ==0) fprintf(screen, "Starting energies are E1 = %f, E2 = %f, E3 = %f.\n", eTLS1, eTLS2, eTLSs);
	double eps = 0.01;
        int me;
        int m;
	int n;
	int dim = domain->dimension;
        MPI_Comm_rank(world,&me);
	double **aVec = atom->x;
	std::cout << "we still in here" << std::endl;
	CopyAtoms(pTLSs, aVec);
	std::cout << "we STILL in here" << std::endl;
        for(int i=0;i<atom->natoms;i++)
        {
		m = atom->map(i);
		std::cout << i << ", " << m << std::endl;
		if (m >= 0 && m < atom->nlocal)
		{
			for(int j=0;j<domain->dimension;j++)
			{
				std::cout << "Shifting: " << aVec[m][j] << ", to " << eps * freqs[2 + dim * i + j] + aVec[m][j] << std::endl;
				aVec[m][j] = eps * freqs[2 + dim * i + j] + aVec[m][j];
			}
		}
        }
	double E1 = CallMinimize();
	if( !CheckMinimum() || (eTLSs - E1) < 0.0) return false;
	CopyAtoms(pTLS1, atom->x);
	CopyBoxToLat(lTLS1);
	eTLS1 = E1;

	CopyAtoms(aVec, pTLSs);
        for(int i=0;i<atom->natoms;i++)
        {
                m = atom->map(i);
                if (m >= 0 && m < atom->nlocal)
                {
                        for(int j=0;j<domain->dimension;j++)
                        {
                                std::cout << "Shifting: " << aVec[m][j] << ", to " << eps * freqs[2 + dim * i + j] + aVec[m][j] << std::endl;
                                aVec[m][j] = eps * freqs[2 + dim * i + j] + aVec[m][j];
                        }
                }
        }
        double E2 = CallMinimize();
        if( !CheckMinimum() || (eTLSs - E2) < 0.0) return false;
        CopyAtoms(pTLS2, atom->x);
        CopyBoxToLat(lTLS2);
        eTLS2 = E2;
	if(comm->me ==0) fprintf(screen, "Ending energies are E1 = %f, E2 = %f, E3 = %f.\n", eTLS1, eTLS2, eTLSs);
	return true;
}
