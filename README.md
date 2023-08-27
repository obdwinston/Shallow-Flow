# Shallow Water Flow Solver

## Case Study

### Monai Valley

![image](https://github.com/obdwinston/Shallow-Flow/assets/104728656/2c45011f-5401-493d-97ed-d96f5b7079f9)

## References

[1] Hou et al. (2015). *An Efficient Unstructured MUSCL Scheme for Solving the 2D Shallow Water Equations.*<br />
[2] Song et al. (2011). *A Robust Well-Balanced Finite Volume Model for Shallow Water Flows with Wetting and Drying over Irregular Terrain.*<br />

## Program Files

| File          | Purpose                                                            |
| :---:         | :---                                                               |
| main.f90      | Main program containing functions and subroutines for flow solver. |
| read.py       | Post-processing program to create depth animation.                 |
| domain.geo  | File containing geometric information to generate mesh in Gmsh.      |
| domain.su2  | File containing mesh information to input to main program.           |

## Program Verification

### Dam Break over Bump

https://github.com/obdwinston/Shallow-Flow/assets/104728656/4e5bcf57-81ba-4598-a16f-ae3fef3bc880

![image](https://github.com/obdwinston/Shallow-Flow/assets/104728656/22c3d069-1dae-40a8-9670-f63e5e859694)

### Circular Dam Break

https://github.com/obdwinston/Shallow-Flow/assets/104728656/c89d26c9-db6b-4bbf-b00d-31623f6a1ac0

![image](https://github.com/obdwinston/Shallow-Flow/assets/104728656/8e15dbc1-484a-4552-be34-df7e1d749fdf)

## Solver Theory

![image](https://github.com/obdwinston/Shallow-Flow/assets/104728656/4a1a5577-734d-4fa3-bbe3-502d50f29a49)
