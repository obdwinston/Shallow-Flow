# Shallow Water Flow Solver

## References

[1] Hou et al. (2015). *An Efficient Unstructured MUSCL Scheme for Solving the 2D Shallow Water Equations.*<br />
[2] Song et al. (2011). *A Robust Well-Balanced Finite Volume Model for Shallow Water Flows with Wetting and Drying over Irregular Terrain.*<br />

## Monai Valley (1993)

For more details on the case study, see [here](https://nctr.pmel.noaa.gov/benchmark/Laboratory/Laboratory_MonaiValley/index.html).

https://github.com/obdwinston/Shallow-Flow/assets/104728656/12dfc506-dc0c-44ca-bcb2-e02e727a548d

https://github.com/obdwinston/Shallow-Flow/assets/104728656/2ed753c9-1193-4728-a0fc-1bb36984d542

![image](https://github.com/obdwinston/Shallow-Flow/assets/104728656/2c45011f-5401-493d-97ed-d96f5b7079f9)

## Program Files

| File          | Purpose                                                                          |
| :---:         | :---                                                                             |
| main.f90      | Main program containing functions and subroutines for flow solver.               |
| read.py       | Post-processing program to create flow animation with Plotly.                    |
| monai.py      | Pre-processing program to set up initial conditions for Monai Valley case study. |
| b_raw.txt     | Raw bathymetry data for Monai Valley case study used in monai.py.                |
| ht_raw.txt    | Raw inflow height data for Monai Valley case study used in monai.py.             |
| monai.geo     | File containing geometric information to generate mesh in Gmsh.                  |
| monai.su2     | File containing mesh information to input to main program.                       |

## Program Verification

### Dam Break over Bump

https://github.com/obdwinston/Shallow-Flow/assets/104728656/4e5bcf57-81ba-4598-a16f-ae3fef3bc880

![image](https://github.com/obdwinston/Shallow-Flow/assets/104728656/8e15dbc1-484a-4552-be34-df7e1d749fdf)

### Circular Dam Break

https://github.com/obdwinston/Shallow-Flow/assets/104728656/c89d26c9-db6b-4bbf-b00d-31623f6a1ac0

![image](https://github.com/obdwinston/Shallow-Flow/assets/104728656/22c3d069-1dae-40a8-9670-f63e5e859694)

## Solver Theory

![image](https://github.com/obdwinston/Shallow-Flow/assets/104728656/4a1a5577-734d-4fa3-bbe3-502d50f29a49)
