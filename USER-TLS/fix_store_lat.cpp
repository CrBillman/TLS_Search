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

#include "stdlib.h"
#include "string.h"
#include "fix_store_lat.h"
#include "domain.h"
#include "atom.h"
#include "memory.h"
#include "error.h"
#include "force.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixStoreLat::FixStoreLat(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
{
  if (narg != 3) error->all(FLERR,"Illegal fix store/lat command");

  // Fix store/lat requires domain box to be created
  if (domain->box_exist == 0)
    error->all(FLERR,"fix store/lat command requires simulation box to be defined");

  // perform initial allocation of 9-element vector to store lattice parameters

  vstore = NULL;
  grow_arrays();

  // zero the storage
  for (int i = 0; i < 9; i++)
    {
      vstore[i] = 0.0;
    }

  // fill vector with lattice parameters
  vstore[0] = domain->boxlo[0];
  vstore[1] = domain->boxlo[1];
  vstore[2] = domain->boxlo[2];
  vstore[3] = domain->boxhi[0];
  vstore[4] = domain->boxhi[1];
  vstore[5] = domain->boxhi[2];
  vstore[6] = domain->xy;
  vstore[7] = domain->xz;
  vstore[8] = domain->yz;
}

/* ---------------------------------------------------------------------- */

FixStoreLat::~FixStoreLat()
{
  memory->destroy(vstore);
}

/* ---------------------------------------------------------------------- */

int FixStoreLat::setmask()
{
  int mask = 0;
  return mask;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double FixStoreLat::memory_usage()
{
  double bytes = 9 * sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   allocate atom-based array
------------------------------------------------------------------------- */

void FixStoreLat::grow_arrays()
{
  memory->grow(vstore,9,"store:vstore");
}

