#!/usr/bin/env python

##
#
# Sanity checks for discrete-time Pontryagin minimum principle. 
#
##

from pydrake.all import *
import numpy as np
import time

def solve_QP(A, B, Q, R, Qf, x_init, T):
    """
    Compute an optimal trajectory by solving the QP
        
        min  sum ( x'Qx + u'Ru ) + x_T'Qfx_T
        s.t. x_{t+1} = Ax + Bu
             x_0 = x_init
    directly. 

    Return the state trajectory
    """
    n = A.shape[1]
    m = B.shape[1]

    # Set up the optimization problem
    mp = MathematicalProgram()
    x = mp.NewContinuousVariables(n, T)
    u = mp.NewContinuousVariables(m, T-1)

    # Initial constraint
    mp.AddConstraint(eq( x[:,0], x_init ))
    for t in range(T-1):
        # Dynamics constraints
        mp.AddConstraint(eq(
            x[:,t+1], A@x[:,t] + B@u[:,t]
        ))

        # Running cost
        mp.AddCost(
            x[:,t].T@Q@x[:,t] + u[:,t].T@R@u[:,t]
        )

    # Terminal cost
    mp.AddCost( x[:,-1].T@Qf@x[:,t] )

    # Solve the optimization problem and extract the solution
    st = time.time()
    res = Solve(mp)
    solve_time = time.time()-st
    print("Solved QP with", res.get_solver_id().name())
    print("Solve time:", solve_time)
    x = res.GetSolution(x)

    return x

def solve_KKT(A, B, Q, R, Qf, x_init, T):
    """
    Compute an optimal trajectory by solving the KKT conditions. 

    Return the state trajectory.
    """
    n = A.shape[1]
    m = B.shape[1]

    # Set up the optimization problem
    mp = MathematicalProgram()
    x = mp.NewContinuousVariables(n, T)
    l = mp.NewContinuousVariables(n, T)    # costate
    u = mp.NewContinuousVariables(m, T-1)

    # Initial condition
    mp.AddConstraint(eq( x[:,0], x_init ))

    for t in range(T-1):
        # forward (state) dynamics
        mp.AddConstraint(eq(
            x[:,t+1], A@x[:,t] + B@u[:,t]
        ))

        # backward (costate) dynamics
        mp.AddConstraint(eq(
            l[:,t], 2*Q@x[:,t] + A.T@l[:,t+1]
        ))

        # Optimal control condition
        mp.AddConstraint(eq(
            0, 2*R@u[:,t] + B.T@l[:,t+1]
        ))

    # Costate boundary condition
    mp.AddConstraint(eq(
        l[:,-1] , 2*Qf@x[:,-1]
    ))

    # Solve the optimization problem and extract the solution
    st = time.time()
    res = Solve(mp)
    solve_time = time.time()-st
    print("Solved KKT with", res.get_solver_id().name())
    print("Solve time:", solve_time)
    x = res.GetSolution(x)

    return x

def solve_PMP(A, B, Q, R, Qf, x_init, T):
    """
    Compute an optimal trajectory by solving the PMP conditions. 

    Return the state trajectory.
    """
    n = A.shape[1]
    m = B.shape[1]

    # Set up the optimization problem
    mp = MathematicalProgram()
    x = mp.NewContinuousVariables(n, T)    # state
    l = mp.NewContinuousVariables(n, T)    # costate

    # Initial condition
    mp.AddConstraint(eq( x[:,0], x_init ))

    for t in range(T-1):
        # Solve for optimal control in terms of costate
        u = -0.5*np.linalg.inv(R)@B.T@l[:,t+1]

        # forward (state) dynamics
        mp.AddConstraint(eq(
            x[:,t+1], A@x[:,t] + B@u
        ))

        # backward (costate) dynamics
        mp.AddConstraint(eq(
            l[:,t], 2*Q@x[:,t] + A.T@l[:,t+1]
        ))

    # Costate boundary condition
    mp.AddConstraint(eq(
        l[:,-1] , 2*Qf@x[:,-1]
    ))

    # Solve the optimization problem and extract the solution
    st = time.time()
    res = Solve(mp)
    solve_time = time.time()-st
    print("Solved PMP with", res.get_solver_id().name())
    print("Solve time:", solve_time)
    x = res.GetSolution(x)

    return x

def plot_state_trajectory(x):
    """
    Make a simple plot of the given state trajectory.
    """
    plt.plot(x.T)
    plt.xlabel("timestep")
    plt.ylabel("state")

if __name__=="__main__":
    # Dynamics
    A = np.array([[1,1],[0,1]])
    B = np.array([[0],[1]])

    # Cost function
    Q = np.eye(2)
    R = 200*np.eye(1)
    Qf = np.eye(2)

    # Initial state
    x_init = np.array([3.,2.])

    # Number of timesteps
    T = 50

    x_QP = solve_QP(A, B, Q, R, Qf, x_init, T)
    x_KKT = solve_KKT(A, B, Q, R, Qf, x_init, T)
    x_PMP = solve_PMP(A, B, Q, R, Qf, x_init, T)

    plt.subplot(3,1,1)
    plot_state_trajectory(x_QP)
    plt.title("QP")
    plt.subplot(3,1,2)
    plot_state_trajectory(x_KKT)
    plt.title("KKT")
    plt.subplot(3,1,3)
    plot_state_trajectory(x_PMP)
    plt.title("PMP")

    plt.show()


