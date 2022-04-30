# crossbar-simulator 

This repo is my current work on simulating neural networks over memristor crossbars. If you're looking at this repo, you probably want to see the crossbar simulation code.

It is in sim/crossbar. crossbar.py contains the bulk of the code for modelling a crossbar. The rest of the files provide utilities and interfaces to the main crossbar solver.

sim/test_system.py is an example of how to use the code.

Here is a brief explanation of the crossbar model:

# Crossbar Model

A resistive array encodes information in the conductances of the crossbar. Voltages are applied to one side and currents are measured at the bottom of the array. The output currents are proportional to the vector matrix product of the device conductances and the input voltage.
There are several nuances to this description:

• Since it is not possible to have a negative conductance in a passive device, a real number is encoded in the difference between neighboring pairs of memristors along the word lines.

• In our crossbar, we do not use analog input voltages. Instead, the output is pieced together
from a binary representation of the input vector.

• The crossbar we use is semi-passive, i.e. it has several 8x8 crossbar arrays instead of one
large passive device or a 1T1R scheme, where each memristor is tied to a transistor.

Since the ideal crossbar is only capable of linear operations, all higher level functions are constructed out of a linear layer, implementing the matrix operation
y ← Ax + b
which is the basic building block of all computational routines. This is implemented in Linear.py.


## Circuit Solver

The crossbar solver implements the routines from [1]. This paper provides equations for the equilibrium solution of a fully passive crossbar array. Because our crossbar is semi-passive, options are given in the device_params dict to construct the array out of smaller tiles. Solving each 8x8 tile involves inverting a 128x128 matrix, since there are two nodes on either side of each memristor and there are 64 memristors. More details are given in the paper.
Crossbar.py loops through the different tiles, applies the inverted matrices, and then sums the output currents. The floating point complexity of the solver is therefore (assuming a square crossbar) O(n^3 * N^2+ n^2 * N^2 * M * d) where

n: tile size \
N: number of tiles \
d: bit precision of the input \
M: number of vmm operations with the same matrix, since inverted tiles are saved by the solver. \

assuming naive matrix inversion and vmm floating point complexity. \
The user interface to the solver involves registering a matrix to the array. The crossbar class returns a ticket object, which has the method vmm which performs the vector matrix multiplication and then the appropriate addition and multiplication to rescale the output.


## Memristor Model

The memristor model is a naive variability scheme in which a memristor is ideally mapped to a target conductance and then perturbed by gaussian noise with standard deviation equal to some percentage of the target conductance. Then a random selection of memristors are placed at the highest and lowest conductance states, imitating stuck-off and stuck-on nonidealities.
Improving the memristor model could be done in two ways:
• Coming up with a better scheme than the naive variability while still using static conducance values.
• Using the circuit solver to time step the state of the entire crossbar. This would be the option for adding VTEAM.


## Adding in VTEAM

The crossbar solver solves the equilibrium voltages and currents of the circuit. To add in a dynamic model like VTEAM which specifies derivatives an ODE solver would be used where the system is solved until steady state is reached, and at that point the output currents are used as the result of the operation.


## References
[1] A Comprehensive Crossbar Array Model With Solutions for Line Resistance and Nonlinear Device Characteristics, An Chen 2013.
