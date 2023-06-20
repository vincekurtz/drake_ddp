## About 

This repository contains code for trajectory optimization using Differential Dynamic Programming (DDP) 
and Iterative LQR (iLQR) in [Drake](https://drake.mit.edu/), with a particular focus on optimization
through [hydroelastic](https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html) contact. 

This code accompanies the paper *Contact-Implicit Trajectory Optimization with Hydroelastic Contact and iLQR*
by Vince Kurtz and Hai Lin, [https://arxiv.org/abs/2202.13986](https://arxiv.org/abs/2202.13986).

## Installation

Install the dependencies:
- Python 3
- Numpy
- [Drake](https://drake.mit.edu/installation.html)

Clone the repository:
```
git clone https://github.com/vincekurtz/drake_ddp/
```

Run the examples (see details below), e.g.,
```
python cart_pole_with_wall.py
```

## Examples

Meldis (`python -m pydrake.visualization.meldis`) must be running to view the generated trajectories. Further parameters can be found in each python script. 

Code for running iLQR over arbitrary discrete-time Drake `System` objects is provided in [`ilqr.py`](ilqr.py). 

### No Contact

These examples are simple benchmark control systems. iLQR can be compared with the direct transcription method for each of these examples: see the parameters in each python script for details. 

[`pendulum.py`](pendulum.py): simple swing-up control of an inverted pendulum.

![](images/pendulum.gif)

[`acrobot.py`](acrobot.py): swingup control of an underactuated acrobot. Performs 50 receding-horizon resolves in the spirit of model predictive control (MPC). 

![](images/acrobot.gif)

[`cart_pole.py`](cart_pole.py): stabilize a cart-pole system around the upright operating point. 

![](images/cart_pole.gif)

### Hydroelastic Contact

These are more complex examples that require making and breaking contact. Contact sequences are determined automatically by iLQR. Drake's AutoDiff capabilities are used to generate dynamics gradients, which can be slow.

[`cart_pole_with_wall.py`](cart_pole_with_wall.py): stabilize a cart-pole system around the upright operating point, with the help of a nearby wall.

![](images/cart_pole_with_wall.gif)

[`kinova_gen3.py`](kinova_gen3.py): Perform whole-arm manipulation of a large ball using a Kinova Gen3 manipulator. 

![](images/kinova.gif)

[`mini_cheetah.py`](mini_cheetah.py): Automatic gait generation for a quadruped robot. Move the MIT Mini Cheetah forward at a desired velocity. Performs 100 receding-horizon resolves in the spirit of MPC. 

![](images/mini_cheetah.gif)

