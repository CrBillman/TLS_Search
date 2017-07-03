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
#include "compute_hessian.h"
#include "pointers.h"
#include "domain.h"
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

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeHessian::ComputeHessian(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg != 4) error->all(FLERR,"Illegal compute hessian command");

  eps = atof(arg[3]); /* atomic shift length in units specified in input*/
  ndof = 3*atom->natoms; /* define number of degrees of freedom */

  array_flag = 1; /* will output 2D Hessian array */
  extarray = 0; /* not extensive quantity */
  size_array_rows = ndof; /* set size of hessian matrix */
  size_array_cols = ndof;
  size_array_rows_variable = 0;

  /* initialize arrays and values */
  hessian = NULL; /* hessian begins as null vector */
  force_ref = NULL; /* reference forces */
  force_ref_loc = NULL; /* reference forces local to each proc*/
  force_shift = NULL; /* forces after coordinate shift */
  force_shift_loc = NULL; /* forces after coordinate shift local to each proc*/
  init();
  allocate(); /* allocate memory for arrays */
}

/* ---------------------------------------------------------------------- */

ComputeHessian::~ComputeHessian()
{
  memory->destroy(hessian);
  memory->destroy(force_ref);
  memory->destroy(force_ref_loc);
  memory->destroy(force_shift);
  memory->destroy(force_shift_loc);
}

/* ---------------------------------------------------------------------- */
void ComputeHessian::init()
{
  int count = 0;
  for (int i = 0; i < modify->ncompute; i++)
    if (strcmp(modify->compute[i]->style,"hessian") == 0) count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR,"More than one compute hessian");

}

/* ---------------------------------------------------------------------- */

void ComputeHessian::compute_array()
{

  /* sets last timestep on which compute_array() invoked */
  invoked_array = update->ntimestep; 
 
  /* Set variables for atomic quantities etc. */
  int *mask = atom->mask;
  int natoms = atom->natoms;
  int nlocal = atom->nlocal;
  int k;
  int k_temp;
  int hess_index1;
  int hess_index2;
  int m;
  int n;

  double mass1; // mass of first atoms
  double mass2; // mass of second atom
  double mass_hess_weight; // mass weighting for hessian element
  double x1orig; // original x position to restore after shift
  double hess_diff; // force difference calculated for hessian element
  double hess_elem; // hessian matrix element

  /* Create pointers to atoms, forces, and masses */
  double **x = atom->x;
  double **f = atom->f;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;

  int vflag = 0; /* vflag (from integrate.h) used for virial computations that we don't need. */
  int eflag = 0; /* same for eflag (don't need energy computation) */

  /* atoms must have consecutive tags for this compute to run correctly. */
  if (atom->tag_enable == 0)
    error->all(FLERR,
               "Hessian compute requires atom IDs");
  if (atom->tag_consecutive() == 0)
    error->all(FLERR,
               "Hessian compute requires that atom IDs are consecutive");

  /* Initialize reference and shifted force arrays */
  for (int i = 0; i < natoms; i++) {
    for (int j = 0; j < domain->dimension; j++) {
      force_ref[3*i+j] = 0.0;
      force_ref_loc[3*i+j] = 0.0;
      force_shift[3*i+j] = 0.0;
      force_shift_loc[3*i+j] = 0.0;
    }
  }

  /* Set up map if none exists (from velocity.cpp line 248) */
  int mapflag = 0;
  if (atom->map_style == 0) {
    mapflag = 1;
    atom->map_init();
    atom->map_set();
  }

 /* allow pair and Kspace compute() to be turned off via modify flags (from integrate.cpp line 52) */
  if (force->pair && force->pair->compute_flag) pair_compute_flag = 1;
  else pair_compute_flag = 0;
  if (force->kspace && force->kspace->compute_flag) kspace_compute_flag = 1;
  else kspace_compute_flag = 0;

  /* Force call to get forces at base configuration (generally taken from verlet.cpp line 300) */
  comm->forward_comm();
  force_clear();
  if (modify->n_pre_force) modify->pre_force(vflag);

  if (pair_compute_flag) force->pair->compute(eflag,vflag);

  if (atom->molecular) {
    if (force->bond) force->bond->compute(eflag,vflag);
    if (force->angle) force->angle->compute(eflag,vflag);
    if (force->dihedral) force->dihedral->compute(eflag,vflag);
    if (force->improper) force->improper->compute(eflag,vflag);
  }

  if (kspace_compute_flag) force->kspace->compute(eflag,vflag);

  // reverse communication of forces if Newton's third law forces
  if (force->newton) comm->reverse_comm();

  if (modify->n_post_force) modify->post_force(vflag);

  // save reference forces using tags to preserve global ID
  // save as 1D array to use MPI_Allreduce and construct hessian easily
  for (int i = 1; i <= natoms; i++) {
    m = atom->map(i);
    if (m >= 0 && m < nlocal) {
      if (mask[m] & groupbit) {
	k = atom->tag[m] - 1;
	for (int j = 0; j < domain->dimension; j++) {    
	  force_ref_loc[3*k+j] = f[m][j];
	}
      }
    }
  }
  // Reduce values of positions and forces and send to all procs
  MPI_Allreduce (force_ref_loc, force_ref, ndof, MPI_DOUBLE, MPI_SUM, world);

  // Loop through all degrees of freedom, shift atomic position, calculate
  // force and corresponding hessian matrix with forward difference

  // OUTER LOOP OVER ATOMS
  for (int i = 1; i <= natoms; i++) {
    m = atom->map(i);
    if (mask[m] & groupbit) {
      hess_index1 = atom->tag[m] - 1;
      MPI_Bcast (&hess_index1, 1, MPI_DOUBLE, comm->me, world);
    
      // Obtain mass of atom i (from compute_ke.cpp)
      if (rmass) {
	mass1 = rmass[m];
	MPI_Bcast (&mass1, 1, MPI_DOUBLE, comm->me, world);
      } else {
	mass1 = mass[type[m]];
	MPI_Bcast (&mass1, 1, MPI_DOUBLE, comm->me, world);
      }
    }
    
    // INNER LOOP OVER DIMENSION - calculate hessian within this loop
    // Add eps shift to atomic coordinate
    for (int j = 0; j < domain->dimension; j++) {
      if (mask[m] & groupbit) {
	x1orig = x[m][j];
	x[m][j] = x[m][j] + eps;
      }
      
      // Force call to get forces after atomic shift
      comm->forward_comm();
      force_clear();
      if (modify->n_pre_force) modify->pre_force(vflag);
      
      if (pair_compute_flag) force->pair->compute(eflag,vflag);
      
      if (atom->molecular) {
	if (force->bond) force->bond->compute(eflag,vflag);
	if (force->angle) force->angle->compute(eflag,vflag);
	if (force->dihedral) force->dihedral->compute(eflag,vflag);
	if (force->improper) force->improper->compute(eflag,vflag);
      }
      
      if (kspace_compute_flag) force->kspace->compute(eflag,vflag);
      
      // Restore original position
      if (mask[m] & groupbit) {
	x[m][j] = x1orig;
      }
      
      // reverse communication of forces if Newton's third law forces
      if (force->newton) comm->reverse_comm();
      if (modify->n_post_force) modify->post_force(vflag);
      
      // Construct force_shift vector and reduce over all procs
      for (int p = 1; p <= natoms; p++) {
	n = atom->map(p);
	k_temp = atom->tag[n] - 1;
	if (n >= 0 && n < nlocal) {
	  if (mask[n] & groupbit) {
	    for (int l = 0; l < domain->dimension; l++) {    
	      force_shift_loc[3*k_temp+l] = f[n][l];
	    }
	  }
	}
      }
      
      // Reduce values of positions and forces and send to all procs
      MPI_Allreduce (force_shift_loc, force_shift, ndof, MPI_DOUBLE, MPI_SUM, world);
      
      // Loop over atoms again to calculate hessian elements
      for (int p = 1; p <= natoms; p++) {
	n = atom->map(p);
	if (mask[n] & groupbit) {
	  hess_index2 = atom->tag[n] - 1;

	  // get second mass for mass-weighting
	  if (rmass) {
	    mass2 = rmass[n];
	  } else {
	    mass2 = mass[type[n]];
	  }
	  mass_hess_weight = 1.0/sqrt(mass1*mass2);

	  // loop over dimensions, calculate hessian element
	  for (int l = 0; l < domain->dimension; l++) {
	    hess_diff = (force_ref[3*hess_index2+l] - force_shift[3*hess_index2+l]);
	    hess_elem = hess_diff*mass_hess_weight*(1.0/eps);
	    hessian[3*hess_index1+j][3*hess_index2+l] = hess_elem;
	  }
	}
      }
    }
  }

  /* destroy the atom map. */
  if (mapflag) {
    atom->map_delete();
    atom->map_style = 0;
  }

  // Standard force call to reset to forces with original positions
  comm->forward_comm();
  force_clear();
  if (modify->n_pre_force) modify->pre_force(vflag);
  
  if (pair_compute_flag) force->pair->compute(eflag,vflag);
  
  if (atom->molecular) {
    if (force->bond) force->bond->compute(eflag,vflag);
    if (force->angle) force->angle->compute(eflag,vflag);
    if (force->dihedral) force->dihedral->compute(eflag,vflag);
    if (force->improper) force->improper->compute(eflag,vflag);
  }
  
  if (kspace_compute_flag) force->kspace->compute(eflag,vflag);
  
  // reverse communication of forces if Newton's third law forces
  if (force->newton) comm->reverse_comm();
  if (modify->n_post_force) modify->post_force(vflag);
  
}


/* ----------------------------------------------------------------------
   clear force on own & ghost atoms
   clear other arrays as needed
   ------------------------------------------------------------------------- */

void ComputeHessian::force_clear()
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

void ComputeHessian::allocate()
{
  memory->destroy(hessian);
  memory->create(hessian,ndof,ndof,"hessian:hessian");
  memory->create(force_ref,3*atom->natoms,"hessian:force_ref");
  memory->create(force_ref_loc,3*atom->natoms,"hessian:force_ref_loc");
  memory->create(force_shift,3*atom->natoms,"hessian:force_shift");
  memory->create(force_shift_loc,3*atom->natoms,"hessian:force_shift_loc");
  array = hessian;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeHessian::memory_usage()
{
  double bytes = ndof * ndof * sizeof(double);
  return bytes;
}
