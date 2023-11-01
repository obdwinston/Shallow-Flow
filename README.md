# Shallow Water Flow Solver

## References

[1] Hou et al. (2015). *An Efficient Unstructured MUSCL Scheme for Solving the 2D Shallow Water Equations.*<br />
[2] Song et al. (2011). *A Robust Well-Balanced Finite Volume Model for Shallow Water Flows with Wetting and Drying over Irregular Terrain.*<br />

## Monai Valley (1993)

The animation shows a numerical simulation of the Shallow Water equations, which approximate free surface shallow flows. The simulation is part of a massive tsunami which engulfed a small offshore island called Okushiri Island, off the west coast of Hokkaido, Japan in 1993. The specific region in the animation is Monai Valley, located on the southwestern coast of Okushiri Island, where an extreme vertical runup of 30 metres was discovered.

Based on the simulation, the maximum vertical runup is approximately 0.08 metres, or 32 metres to scale, which closely matches the field observation. It never ceases to amaze me how, despite simplifying assumptions on the full governing equations (i.e. Navier-Stokes) and approximating (albeit clever) numerical schemes, the numerical solution is still able to match up to physical experiments and field observations.

The numerical scheme was referenced from Hou et al. (2015) and coded in Fortran for the main solver. Pre- and post-processing programs are coded in Python. For more details on the case study, see [here](https://nctr.pmel.noaa.gov/benchmark/Laboratory/Laboratory_MonaiValley/index.html).

https://github.com/obdwinston/Shallow-Flow/assets/104728656/3e750be1-55ad-4df7-8f9e-2e18c5a09010

https://github.com/obdwinston/Shallow-Flow/assets/104728656/597c04c6-8a13-4868-8d89-f7164bb3039d

![image](https://github.com/obdwinston/Shallow-Flow/assets/104728656/2c45011f-5401-493d-97ed-d96f5b7079f9)

## Program Files

| File          | Purpose                                                                          |
| :---:         | :---                                                                             |
| main.f90      | Main program containing functions and subroutines for flow solver.               |
| monai.py      | Pre-processing program to set up initial conditions for Monai Valley case study.<br />*Run this whenever (1) running case study for the first time, (2) there is a change in mesh size, or (3) there is a change in time step.*|
| b_raw.txt     | Raw bathymetry data for Monai Valley case study used in monai.py.                |
| ht_raw.txt    | Raw inflow height data for Monai Valley case study used in monai.py.             |
| read.py       | Post-processing program to create flow animation with Plotly.                    |
| monai.geo     | File containing geometric information to generate mesh in Gmsh.                  |
| monai.su2     | File containing mesh information to input to main program.                       |

## Program Verification

### Dam Break over Bump

https://github.com/obdwinston/Shallow-Flow/assets/104728656/6a54d96f-0595-4afd-a1e3-0928f3d857eb

![image](https://github.com/obdwinston/Shallow-Flow/assets/104728656/8e15dbc1-484a-4552-be34-df7e1d749fdf)

### Circular Dam Break

https://github.com/obdwinston/Shallow-Flow/assets/104728656/087ef536-54d3-49ce-bf0b-8bd194446bcb

![image](https://github.com/obdwinston/Shallow-Flow/assets/104728656/22c3d069-1dae-40a8-9670-f63e5e859694)

## Solver Theory

![image](https://github.com/obdwinston/Shallow-Flow/assets/104728656/4a1a5577-734d-4fa3-bbe3-502d50f29a49)
