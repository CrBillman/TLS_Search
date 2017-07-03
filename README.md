This package contains the machinery to perform a two-level system (TLS) search.
It combines bisection, ridge, and couple_elastic commands, ad well as 
compute hessian and compute freq commands, and a fix store to store the lattice to expand
the TLS search to NPT simulations.

To use, type 'make yes-user-tls' from the src directory.  Then, make
lammps as you normally would for your system.

The people who created this package are Chris Billman (cbillman1117@gmail.com) and Jonathan Trinastic
(jptrinastic at gmail dot com) at the University of Florida in Prof. Hai-Ping Cheng's research group. Contact
them directly if you have questions.

*************************************************************************************************************

The goal of the TLS search is to sample the potential energy landscape and locate adjacent minima that are connected
through a single saddle point.  More information can be found in R Hamdan, JP Trinastic, HP Cheng.
The Journal of chemical physics 141 (5), 054501; JP Trinastic, R Hamdan, C Billman, HP Cheng.
Physical Review B 93 (1), 014105; CR Billman, JP Trinastic, DJ Davis, R Hamdan, HP Cheng.
Physical Review B 95 (1), 014109.  In this package, the search is performed by running trajectories in classical MD.  
The bisection method locates adjacent minima from the trajectory, and also evaluates which steps in the trajectory 
are closest while still spanning the ridge between the two minima. The ridge method takes this information, and 
locates the saddle point between the minima.  If the saddle point is successfully located, the asymmetry and barrier
height is calculated along with the relaxation time of each minima and the RMS distance between the wells.  After 
the TLS has been located, the coupling constant of the TLS and the elastic moduli of each minima configuration 
are calculated.  The TLS then continues by running another trajectory.

Because of the disoder in amorphous materials, the TLS search is inherently stochastic.  To obtain converged
properties, the search must be run for long times and for many different configurations of a material.  This
also creates problems when implementing a TLS search, as there are some parameters that need to be altered for the
search to be effective in a material.  Most of these parameters are input parameters for the main methods of the
TLS search.  In general, these parameters should be tested for each material.  For tantala and silica, we have
found that these parameters result in good performance.  To evaluate performance, the success rate of the
bisection and ridge methods should be evaluated.  All of the output of the TLS search is written to the output and
preceded by the string "UPDATE".  To diagnose problems and optimize parameters, it is essential that the TLS Search
output is investigated.  The easiest way to do this is to grep UPDATE <output> > updates and examine the output there.

Commands:

bisection nSteps dCut mStore file
    nSteps: The number of steps in the trajectory.  A good value is 1500.
    dCut: A cut-off distance for minima comparison.  To determine if two minima are identical, the RMS distance between 
        the wells is computed.  If this distance is smaller than dCut, the minima are labelled identical.  A good value 
        is 0.02 Angstroms.
    mStore: The method of storing the trajectory.  Currently, it's stored in a dump file, and the corresponding option 
        is "FMD".
    file: The location of the trajectory.  Example scripts store the trajectory in dump.min.

ridge maxLoops nBisection dCut maxForce
    maxLoops: The maximum number of iterations of the ridge method.  A good value is 100.
    nRelaxation: The number of times that configurations are bisected during the ridge method. A good value is 3.
    dCut: A cut-off distance for minima comparison.  To determine if two minima are identical, the RMS distance between
        the wells is computed.  If this distance is smaller than dCut, the minima are labelled identical.  A good value 
        is 0.02 Angstroms.
    maxForce: The force tolerance for the saddle point.  A good value is 15e-3.

couple_elastic maxStrain mSlope
    maxStrain: The maximum strain to apply to the unit cell when calculating the coupling constant and elastic moduli.
        This value should be small enough not to violate the linear response approximation.  A good value is 0.005.
    mSlope: The method of calculating the slope from the evaluated pressure and energy.  Currently, only "linear" is supported.

Two examples are included with this package.  In the first, the bisection and ridge method are demonstrated on an SiO2 "molecule".
The oxygen atoms are fixed 10 Angstroms away, so that the Si atom cannot bond to both simultaneously.  As the Si atom moves betwen
them, it moves through a saddle point.  This script locates the two minima and saddle point between the O atoms.

In the second example, the TLS search is run on an amorphous silica sample.  The sample contains 644 atoms.  The example
illustrates the difficulty of finding a TLS, as it takes 12 loops to locate a TLS in the search.

When a TLS is successfully located, its properties can be located in TLS.dump, TLS.pos, and cc_ec_final.dump.  The formats of
these files is explained below:

TLS.dump contains the output from both the bisection and ridge methods.  When the bisection method finishes successfully, it writes
Bisection <Asymmetry> <RMS distance>

When the ridge method finishes successfully, it writes
<Asymmetry> <Barrier Height> <RMS distance> <Barrier Height from the left> <Barrier Height from the right> <Left Relaxation Time> <Right Relaxation Time>

When a TLS is successfully located, the atomic positions of each minima and the saddle point is dumped to TLS.pos.  The timesteps identify
which configuration it is.  For the first minima, the timestep is 0; for the second minima, the timestep is 1; for the saddle point, the timestep is 2.
The rest of the file follows LAMMPS dump formatting.

The output of the couple_elastic calculation is stored in cc_ec_final.dump.  A header is printed each time the calculation is run,
which labels each output.
