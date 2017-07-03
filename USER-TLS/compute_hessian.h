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

#ifdef COMPUTE_CLASS

ComputeStyle(hessian,ComputeHessian)

#else

#ifndef LMP_COMPUTE_HESSIAN_H
#define LMP_COMPUTE_HESSIAN_H

#include "compute.h"

namespace LAMMPS_NS {

  class ComputeHessian : public Compute {
  public:
    ComputeHessian(class LAMMPS *, int, char **);
    ~ComputeHessian();
    void init();
    void compute_array(); 
    void allocate();
    double memory_usage();

    
  protected:
    double eps; // shift in atomic position in distance units
    int ndof; // degrees of freedom - hessian length
    double ** hessian; // hessian matrix
    double * force_ref; // forces for base configuration
    double * force_ref_loc; // local forces for base configuration
    double * force_shift; // forces for shifted configuration
    double * force_shift_loc; // local forces for shifted configuration   
    int pair_compute_flag; // 0 if pair->compute is skipped
    int kspace_compute_flag; // 0 if kspace->compute is skipped
    
    void force_clear(); // Clear current force array
  };
  
}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

W: More than one compute hessian command

*/
