#!/usr/bin/env python

##
#
# Do swing-up control of an acrobot
#
##

import numpy as np
from pydrake.all import *
from ilqr import IterativeLinearQuadraticRegulator
import time

####################################
# Parameters
####################################

T = 1.5        # total simulation time (S)
dt = 1e-2      # simulation timestep

# Solver method
# must be "ilqr" or "sqp"
method = "ilqr"
MPC = True      # MPC only works with ilqr for now

# Initial state
x0 = np.array([0,0,0,0])

# Target state
x_nom = np.array([np.pi,0,0,0])

# Quadratic cost int_{0^T} (x'Qx + u'Ru) + x_T*Qf*x_T
Q = 0.01*np.diag([0,0,1,1])
R = 0.01*np.eye(1)
Qf = 100*np.diag([1,1,1,1])

####################################
# Tools for system setup
####################################

def create_system_model(plant):
    urdf = FindResourceOrThrow("drake/examples/acrobot/Acrobot.urdf")
    robot = Parser(plant=plant).AddModelFromFile(urdf)
    plant.Finalize()
    return plant

####################################
# Create system diagram
####################################
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, dt)
plant = create_system_model(plant)
assert plant.geometry_source_is_registered()

controller = builder.AddSystem(ConstantVectorSource(np.zeros(1)))
builder.Connect(
        controller.get_output_port(),
        plant.get_actuation_input_port())

DrakeVisualizer().AddToBuilder(builder, scene_graph)
ConnectContactResultsToDrakeVisualizer(builder, plant, scene_graph)

diagram = builder.Build()
diagram_context = diagram.CreateDefaultContext()
plant_context = diagram.GetMutableSubsystemContext(
        plant, diagram_context)

#####################################
# Solve Trajectory Optimization
#####################################

# System model for the trajectory optimizer
plant_ = MultibodyPlant(dt)
plant_ = create_system_model(plant_)
input_port_index = plant_.get_actuation_input_port().get_index()

#-----------------------------------------
# DDP method
#-----------------------------------------

def solve_ilqr(solver, x0, u_guess):
    """
    Convienience function for solving the optimization
    problem from the given initial state with the given
    guess of control inputs.
    """
    solver.SetInitialState(x0)
    solver.SetInitialGuess(u_guess)

    states, inputs, solve_time, optimal_cost = solver.Solve()
    return states, inputs, solve_time, optimal_cost

if method == "ilqr":
    # Set up optimizer
    num_steps = int(T/dt)
    ilqr = IterativeLinearQuadraticRegulator(plant_, num_steps, 
            input_port_index=input_port_index,
            beta=0.5)

    # Define the problem
    ilqr.SetTargetState(x_nom)
    ilqr.SetRunningCost(dt*Q, dt*R)
    ilqr.SetTerminalCost(Qf)

    # Set initial guess
    u_guess = np.zeros((1,num_steps-1))


    if MPC:
        # MPC parameters
        num_resolves = 50    # total number of times to resolve the optimizaiton problem
        replan_steps = 2     # number of timesteps after which to move the horizon and
                             # re-solve the MPC problem (>0)
        total_num_steps = num_steps + replan_steps*num_resolves
        total_T = total_num_steps*dt

        states = np.zeros((4,total_num_steps))

        # Solve to get an initial trajectory
        x, u, _, _ = solve_ilqr(ilqr, x0, u_guess)
        states[:,0:num_steps] = x

        for i in range(num_resolves+1):
            # Set new state and control input
            last_u = u[:,-1]
            u_guess = np.block([
                u[:,replan_steps:],    # keep same control inputs from last optimal sol'n
                np.repeat(last_u[np.newaxis].T,replan_steps,axis=1)  # for new timesteps copy
                ])                                                   # the last known control input
            x0 = x[:,replan_steps]

            # Resolve the optimization
            x, u, _, _ = solve_ilqr(ilqr, x0, u_guess)

            # Save the result for playback
            start_idx = i*replan_steps
            end_idx = start_idx + num_steps
            states[:,start_idx:end_idx] = x

        timesteps = np.arange(0.0,total_T,dt)
    else:
        states, inputs, solve_time, optimal_cost = solve_ilqr(ilqr, x0, u_guess)
        print(f"Solved in {solve_time} seconds using iLQR")
        print(f"Optimal cost: {optimal_cost}")
        timesteps = np.arange(0.0,T,dt)

#-----------------------------------------
# Direct Transcription method
#-----------------------------------------

elif method == "sqp":
    context_ = plant_.CreateDefaultContext()

    # Set up the solver object
    trajopt = DirectTranscription(
            plant_, context_, 
            input_port_index=input_port_index,
            num_time_samples=int(T/dt))
    
    # Add constraints
    x = trajopt.state()
    u = trajopt.input()
    x_init = trajopt.initial_state()
    
    trajopt.prog().AddConstraint(eq( x_init, x0 ))
    x_err = x - x_nom
    trajopt.AddRunningCost(x_err.T@Q@x_err + u.T@R@u)
    trajopt.AddFinalCost(x_err.T@Qf@x_err)
    
    # Solve the optimization problem
    st = time.time()
    res = Solve(trajopt.prog())
    solve_time = time.time() - st
    assert res.is_success(), "trajectory optimizer failed"
    solver_name = res.get_solver_id().name()
    optimal_cost = res.get_optimal_cost()
    print(f"Solved in {solve_time} seconds using SQP via {solver_name}")
    print(f"Optimal cost: {optimal_cost}")
    
    # Extract the solution
    timesteps = trajopt.GetSampleTimes(res)
    states = trajopt.GetStateSamples(res)
    inputs = trajopt.GetInputSamples(res)

else:
    raise ValueError("Unknown method %s"%method)

#####################################
# Playback
#####################################

def playback(states, timesteps):
    """
    Convienience function for visualising the given trajectory.

    Relies on diagram, diagram_context, plant, and plant_context
    being defined outside of the scope of this function and connected
    to the Drake visualizer.
    """
    while True:
        # Just keep playing back the trajectory
        for i in range(len(timesteps)):
            t = timesteps[i]
            x = states[:,i]

            diagram_context.SetTime(t)
            plant.SetPositionsAndVelocities(plant_context, x)
            diagram.Publish(diagram_context)

            time.sleep(dt-3e-4)
        time.sleep(1)

playback(states,timesteps)
