#!/usr/bin/env python

##
#
# Swing-up control of a simple inverted pendulum
#
##

import numpy as np
from pydrake.all import *
from ilqr import IterativeLinearQuadraticRegulator
from pontryagin import PontryaginOptimizer
import time

####################################
# Parameters
####################################

T = 1.0        # total simulation time (S)
dt = 1e-2      # simulation timestep

# Solver method
# must be "ilqr" or "sqp" or "pontryagin"
method = "pontryagin"

# Initial state
x0 = np.array([0.0,0.0])

# Target state
x_nom = np.array([0.5,0])

# Quadratic cost int_{0^T} (x'Qx + u'Ru) + x_T*Qf*x_T
Q = 0.01*np.diag([0,1])
R = 0.01*np.eye(1)
Qf = 100*np.diag([1,1])

####################################
# Tools for system setup
####################################

def create_system_model(builder, dt):
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, dt)

    mass = 1.0
    radius = 0.01
    length = 0.5
    damping = 0.0

    # Create the pendulum object
    pendulum = plant.AddModelInstance("pendulum")
    rod = plant.AddRigidBody("rod", pendulum, SpatialInertia(mass, [0,0,0],
                UnitInertia.SolidCylinder(radius, length)))
    rod_com_frame = plant.GetFrameByName("rod")
    X_base_com = RigidTransform(RotationMatrix(),[0,0,length/2])
    rod_base_frame = plant.AddFrame(FixedOffsetFrame("rod_base",
        rod_com_frame, X_base_com))
    #base_joint = plant.AddJoint(RevoluteJoint("base_joint", plant.world_frame(),
    #    rod_base_frame, [1,0,0], damping))
    base_joint = plant.AddJoint(PrismaticJoint("base_joint", plant.world_frame(),
        rod_base_frame, [1,0,0], damping))
    plant.AddJointActuator("base_actuator", base_joint)
    rod_shape = Cylinder(radius, length)
    plant.RegisterVisualGeometry(rod, RigidTransform(), rod_shape, "rod_visual", 
            [0.5,0.5,0.9,1])

    # Add a sphere on the end of the pendulum
    ball_shape = Sphere(2*radius)
    ball = plant.AddRigidBody("ball", pendulum, SpatialInertia(mass, [0,0,0],
                UnitInertia.SolidSphere(2*radius)))
    X_ball = RigidTransform(RotationMatrix(),[0,0,-length/2])
    plant.WeldFrames(plant.GetFrameByName("rod"), plant.GetFrameByName("ball"),
            X_ball)
    plant.RegisterVisualGeometry(ball, RigidTransform(), ball_shape,
            "ball_visual", [0.5,0.5,0.9,1])

    plant.Finalize()
    plant.set_name("plant")

    return plant, scene_graph

####################################
# Create system diagram
####################################

builder = DiagramBuilder()

plant, scene_graph = create_system_model(builder, dt)
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

# Create system model for the solver to use
builder_ = DiagramBuilder()
plant_, scene_graph_ = create_system_model(builder_, dt)
builder_.ExportInput(plant_.get_actuation_input_port(),"control")
system_ = builder_.Build()

#-----------------------------------------
# iLQR method
#-----------------------------------------

if method == "ilqr":
    # Set up the optimizer
    num_steps = int(T/dt)
    ilqr = IterativeLinearQuadraticRegulator(system_, num_steps, 
            autodiff=True)

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
# Pontryagin method (experimental)
#-----------------------------------------

elif method == "pontryagin":
    # Set up the optimizer
    num_steps = int(T/dt)
    popt = PontryaginOptimizer(system_, num_steps)

    # Define initial and target states
    popt.SetInitialState(x0)
    popt.SetTargetState(x_nom)

    # Define cost function
    popt.SetRunningCost(dt*Q, dt*R)
    popt.SetTerminalCost(Qf)

    # Set initial guess
    u_guess = np.zeros((1,num_steps-1))
    x_guess = np.zeros((2,num_steps))
    lambda_guess = np.zeros((2,num_steps))  # costate
    popt.SetInitialGuess(x_guess, u_guess, lambda_guess)

    states, inputs, solve_time, optimal_cost = popt.Solve()
    print(f"Solved in {solve_time} seconds using special Pontryagin method")
    print(f"Optimal cost: {optimal_cost}")
    timesteps = np.arange(0.0,T,dt)

#-----------------------------------------
# Direct Transcription method
#-----------------------------------------

elif method == "sqp":
    context_ = plant_.CreateDefaultContext()
    input_port_index = plant_.get_actuation_input_port().get_index()

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
    prog = trajopt.prog()
    prog.SetSolverOption(SolverType.kSnopt, "Print file", "/tmp/snopt.out")
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
