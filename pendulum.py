#!/usr/bin/env python

##
#
# Do swing-up control of a simple inverted pendulum
#
##

import numpy as np
from pydrake.all import *
from ilqr import IterativeLinearQuadraticRegulator
import time

####################################
# Parameters
####################################

T = 2.0         # total simulation time (S)
dt = 1e-2      # simulation timestep

# Solver method
# must be "ilqr" or "sqp"
method = "ilqr"

# Initial state
x0 = np.array([0,0])

# Target state
x_nom = np.array([np.pi,0])

# Quadratic cost int_{0^T} (x'Qx + u'Ru) + x_T*Qf*x_T
Q = 0.01*np.diag([0,1])
R = 0.01*np.eye(1)
Qf = 100*np.diag([1,1])

####################################
# Tools for system setup
####################################

def create_system_model(plant):
    urdf = FindResourceOrThrow("drake/examples/pendulum/Pendulum.urdf")
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
# Create system model
plant_ = MultibodyPlant(dt)
plant_ = create_system_model(plant_).ToAutoDiffXd()

builder_ = DiagramBuilder_[AutoDiffXd]()
plant_, scene_graph_ = AddMultibodyPlantSceneGraph(builder_, plant_)
diagram_ = builder_.Build()
diagram_context_ = diagram_.CreateDefaultContext()
context_ = diagram_.GetMutableSubsystemContext(plant_, diagram_context_)

#-----------------------------------------
# DDP method
#-----------------------------------------

if method == "ilqr":
    # Set up the optimizer
    num_steps = int(T/dt)
    ilqr = IterativeLinearQuadraticRegulator(plant_, context_, num_steps)

    # Define initial and target states
    ilqr.SetInitialState(x0)
    ilqr.SetTargetState(x_nom)

    # Define cost function
    ilqr.SetRunningCost(dt*Q, dt*R)
    ilqr.SetTerminalCost(Qf)

    # Set initial guess
    u_guess = np.zeros((1,num_steps-1))
    ilqr.SetInitialGuess(u_guess)

    states, inputs, solve_time, optimal_cost = ilqr.Solve()
    print(f"Solved in {solve_time} seconds using iLQR")
    print(f"Optimal cost: {optimal_cost}")
    timesteps = np.arange(0.0,T,dt)

#-----------------------------------------
# Direct Transcription method
#-----------------------------------------

elif method == "sqp":
    # Convert to scalar type type
    plant_ = plant_.ToScalarType[float]()
    context_ = plant_.CreateDefaultContext()

    # Set up the solver object
    trajopt = DirectTranscription(
            plant_, context_, 
            input_port_index=plant.get_actuation_input_port().get_index(),
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
