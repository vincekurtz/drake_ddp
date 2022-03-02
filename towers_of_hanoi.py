#!/usr/bin/env python

##
#
# Contact-implicit trajectory optimization for whole-arm manipulation
# using a Kinova Gen3 manipulator arm.
#
##

import numpy as np
from pydrake.all import *
from ilqr import IterativeLinearQuadraticRegulator

# Choose what to do
simulate = True   # Run a simple simulation with fixed input
optimize = False    # Find an optimal trajectory using ilqr
playback = False    # Visualize the optimal trajectory by playing it back.
                   # If optimize=False, attempts to load a previously saved
                   # trajectory from a file.

save_file = "data/towers_of_hanoi.npz"

####################################
# Parameters
####################################

T = 0.005
dt = 1e-2
playback_rate = 0.5

# Some useful configuration space definitions
q_start = np.array([0.0,0.0,0.2, 0.0,0.0,0.0])   # x y z roll pitch yaw

# Initial state
x0 = np.hstack([q_start, np.zeros(6)])

# Target state
x_nom = np.hstack([q_start, np.zeros(6)])

# Quadratic cost
Q = np.eye(13)
R = 0.01*np.eye(6)
Qf = np.eye(13)

# Contact model parameters
dissipation = 1.0              # controls "bounciness" of collisions: lower is bouncier
hydroelastic_modulus = 5e6     # controls "squishiness" of collisions: lower is squishier
resolution_hint = 0.05         # smaller means a finer mesh

mu_static = 0.6
mu_dynamic = 0.5

contact_model = ContactModel.kHydroelastic  # Hydroelastic, Point, or HydroelasticWithFallback
mesh_type = HydroelasticContactRepresentation.kPolygon  # Triangle or Polygon

####################################
# Tools for system setup
####################################

def create_system_model(plant, scene_graph):
    # Add an unactuated gripper from urdf
    # (rigid hydroelastic contact included)
    urdf = "models/2f_85_gripper/urdf/robotiq_85_gripper.urdf"
    gripper = Parser(plant).AddModelFromFile(urdf)

    # Add a ground with compliant hydroelastic contact
    ground_props = ProximityProperties()
    AddCompliantHydroelasticProperties(resolution_hint, hydroelastic_modulus,ground_props)
    friction = CoulombFriction(0.7*mu_static, 0.7*mu_dynamic)
    AddContactMaterial(dissipation=dissipation, friction=friction, properties=ground_props)
    X_ground = RigidTransform()
    X_ground.set_translation([0,0,-0.5])
    ground_shape = Box(25,25,1)
    plant.RegisterCollisionGeometry(plant.world_body(), X_ground,
            ground_shape, "ground_collision", ground_props)


    # Choose contact model
    plant.set_contact_surface_representation(mesh_type)
    plant.set_contact_model(contact_model)
    plant.Finalize()

    return plant, scene_graph

####################################
# Create system diagram
####################################
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, dt)
plant, scene_graph = create_system_model(plant, scene_graph)

# Connect to visualizer
params = DrakeVisualizerParams(role=Role.kIllustration, show_hydroelastic=True)
DrakeVisualizer(params=params).AddToBuilder(builder, scene_graph)
ConnectContactResultsToDrakeVisualizer(builder, plant, scene_graph)

# Finailze the diagram
diagram = builder.Build()
diagram_context = diagram.CreateDefaultContext()
plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)

####################################
# Solve Trajectory Optimization
####################################

if optimize:
    # Create a system model (w/o visualizer) to do the optimization over
    builder_ = DiagramBuilder()
    plant_, scene_graph_ = AddMultibodyPlantSceneGraph(builder_, dt)
    plant_, scene_graph_ = create_system_model(plant_, scene_graph_)
    builder_.ExportInput(plant_.get_actuation_input_port(), "control")
    system_ = builder_.Build()

    # Set up the optimizer
    num_steps = int(T/dt)
    ilqr = IterativeLinearQuadraticRegulator(system_, num_steps, 
            beta=0.5, delta=1e-3, gamma=0)

    # Define the optimization problem
    ilqr.SetInitialState(x0)
    ilqr.SetTargetState(x_nom)
    ilqr.SetRunningCost(dt*Q, dt*R)
    ilqr.SetTerminalCost(Qf)

    # Set initial guess
    plant.SetPositionsAndVelocities(plant_context, x0)
    tau_g = -plant.CalcGravityGeneralizedForces(plant_context)
    S = plant.MakeActuationMatrix().T
    u_gravity_comp = S@np.repeat(tau_g[np.newaxis].T, num_steps-1, axis=1)

    #u_guess = np.zeros((plant.num_actuators(),num_steps-1))
    u_guess = u_gravity_comp
    ilqr.SetInitialGuess(u_guess)

    # Solve the optimization problem
    states, inputs, solve_time, optimal_cost = ilqr.Solve()
    print(f"Solved in {solve_time} seconds using iLQR")
    print(f"Optimal cost: {optimal_cost}")
    timesteps = np.arange(0.0,T,dt)

    # save the solution
    ilqr.SaveSolution(save_file)

#####################################
# Playback
#####################################

if playback:

    if not optimize:
        # load previously computed solution from file
        data = np.load(save_file)
        timesteps = data["t"]
        states = data["x_bar"]

    while True:
        plant.get_actuation_input_port().FixValue(plant_context, 
                np.zeros(plant.num_actuators()))
        # Just keep playing back the trajectory
        for i in range(len(timesteps)):
            t = timesteps[i]
            x = states[:,i]

            diagram_context.SetTime(t)
            plant.SetPositionsAndVelocities(plant_context, x)
            diagram.Publish(diagram_context)

            time.sleep(1/playback_rate*dt-4e-4)
        time.sleep(1)

####################################
# Run Simulation
####################################

if simulate:
    # Fix zero input for now
    plant.get_actuation_input_port().FixValue(plant_context, np.zeros(plant.num_actuators()))

    # Set initial state
    #plant.SetPositionsAndVelocities(plant_context, x0)
    print(plant.num_positions())
    print(plant.num_velocities())
    print(plant.num_actuators())

    print(plant.CalcTotalMass(plant_context))
    
    # Simulate the system
    simulator = Simulator(diagram, diagram_context)
    simulator.set_target_realtime_rate(playback_rate)
    simulator.set_publish_every_time_step(True)

    simulator.AdvanceTo(T)
