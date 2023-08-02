#!/usr/bin/env python

##
#
# Swing-up control of a cart-pole system
#
##

import numpy as np
from pydrake.all import *
from ilqr import IterativeLinearQuadraticRegulator
import time
import utils_derivs_interpolation

meshcat_visualisation = True

####################################
# Parameters
####################################

T = 2.0        # total simulation time (S)
dt = 0.01      # simulation timestep

# Parameters for derivative interpolation
keypoint_method = 'adaptiveJerk'        # 'setInterval, or 'adaptiveJerk' or 'iterativeError'
minN = 5                                # Minimum interval between key-points   
maxN = 10                               # Maximum interval between key-points
jerk_threshold = 1e-4                   # Jerk threshold to trigger new key-point (only used in adaptiveJerk)
iterative_error_threshold = 0.00005     # Error threshold to trigger new key-point (only used in iterativeError)

# Solver method
# must be "ilqr" or "sqp"
method = "ilqr"

# Initial state
# x0 = np.array([0,np.pi-0.1,0,0])
x0 = np.array([0,0,0,0])

# Target state
x_nom = np.array([0,np.pi,0,0])

# Quadratic cost int_{0^T} (x'Qx + u'Ru) + x_T*Qf*x_T
Q = np.diag([10,10,0.1,0.1])
R = 0.001*np.eye(1)
Qf = np.diag([100,100,10,10])

####################################
# Tools for system setup
####################################

def create_system_model(plant):
    sdf = FindResourceOrThrow("drake/examples/multibody/cart_pole/cart_pole.sdf")
    robot = Parser(plant=plant).AddModelFromFile(sdf)
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

if meshcat_visualisation:
    meshcat = StartMeshcat()
    visualizer = MeshcatVisualizer.AddToBuilder( 
        builder, scene_graph, meshcat,
        MeshcatVisualizerParams(role=Role.kPerception, prefix="visual"))
else:
    DrakeVisualizer().AddToBuilder(builder, scene_graph)
    ConnectContactResultsToDrakeVisualizer(builder, plant, scene_graph)

diagram = builder.Build()
diagram_context = diagram.CreateDefaultContext()
plant_context = diagram.GetMutableSubsystemContext(
        plant, diagram_context)

#####################################
# Solve Trajectory Optimization
#####################################
    
# Create system model for controller
plant_ = MultibodyPlant(dt)
plant_ = create_system_model(plant_)
input_port_index = plant_.get_actuation_input_port().get_index()

#-----------------------------------------
# DDP method
#-----------------------------------------

if method == "ilqr":
    # Set up the optimizer
    num_steps = int(T/dt)
    interpolation_method = utils_derivs_interpolation.derivs_interpolation(keypoint_method, minN, maxN, jerk_threshold, iterative_error_threshold)
    ilqr = IterativeLinearQuadraticRegulator(plant_, num_steps, 
            input_port_index=input_port_index,
            beta=0.9, derivs_keypoint_method=interpolation_method)

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

while True:
    # Just keep playing back the trajectory
    for i in range(len(timesteps)):
        t = timesteps[i]
        x = states[:,i]

        diagram_context.SetTime(t)
        plant.SetPositionsAndVelocities(plant_context, x)
        diagram.ForcedPublish(diagram_context)

        time.sleep(dt-3e-4)
    time.sleep(1)
