# Shallow Water Flow Solver

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

https://github.com/obdwinston/Shallow-Flow/assets/104728656/77c3774e-712e-4828-ad7e-03762de29aa9

![image](https://github.com/obdwinston/Shallow-Flow/assets/104728656/5ad75df6-aac1-4123-830a-c3429165328f)

### Circular Dam Break

https://github.com/obdwinston/Shallow-Flow/assets/104728656/6bbf543d-caae-4997-88a7-d0ebe141e2b5

![image](https://github.com/obdwinston/Shallow-Flow/assets/104728656/6da236e6-44d5-4403-a414-3aa82b754f6f)

## Solver Theory

![image](https://github.com/obdwinston/Shallow-Flow/assets/104728656/4a1a5577-734d-4fa3-bbe3-502d50f29a49)
