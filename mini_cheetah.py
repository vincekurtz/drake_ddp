#!/usr/bin/env python

##
#
# Simple simulation to test Drake's new hydroelastic 
# contact model on the mini cheetah quadruped.
#
##

import numpy as np
from pydrake.all import *
from ilqr import IterativeLinearQuadraticRegulator

####################################
# Parameters
####################################

T = 0.5
dt = 1e-2
playback_rate = 0.2

# Some useful joint angle definitions
q0 = np.asarray([ 1.0, 0.0, 0.0, 0.0,     # base orientation
                  0.0, 0.0, 0.3,          # base position
                  0.0,-0.8, 1.6,
                  0.0,-0.8, 1.6,
                  0.0,-0.8, 1.6,
                  0.0,-0.8, 1.6])

# Initial state
x0 = np.hstack([q0, np.zeros(18)])

# Target state
x_nom = x0

# Quadratic cost
Qq_robot = 0.0*np.ones(7)
Qv_robot = 1.0*np.ones(7)
Qq_ball = np.array([0,0,0,0,100,100,100])
Qv_ball = np.ones(6)
Q_diag = np.hstack([Qq_robot, Qq_ball, Qv_robot, Qv_ball])
Qf_diag = np.hstack([Qq_robot, Qq_ball, Qv_robot, 5*Qv_ball])

Q = np.diag(Q_diag)
R = 0.01*np.eye(7)
Qf = np.diag(Qf_diag)

# Contact model parameters
contact_model = ContactModel.kHydroelastic  # Hydroelastic, Point, or HydroelasticWithFallback
mesh_type = HydroelasticContactRepresentation.kPolygon  # Triangle or Polygon

####################################
# Tools for system setup
####################################

def create_system_model(plant):

    # Add the kinova arm model from urdf (compliant hydroelastic contact included)
    urdf = "models/mini_cheetah/mini_cheetah_mesh.urdf"
    arm = Parser(plant).AddModelFromFile(urdf)

    # Add a ground with rigid hydroelastic contact
    ground_props = ProximityProperties()
    AddRigidHydroelasticProperties(ground_props)
    AddContactMaterial(friction=CoulombFriction(), properties=ground_props)
    plant.RegisterCollisionGeometry(plant.world_body(), RigidTransform(),
            HalfSpace(), "ground_collision", ground_props)

    # Choose contact model
    plant.set_contact_surface_representation(mesh_type)
    plant.set_contact_model(contact_model)
    plant.Finalize()

    return plant

####################################
# Create system diagram
####################################
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, dt)
plant = create_system_model(plant)

# Connect to visualizer
DrakeVisualizer().AddToBuilder(builder, scene_graph)
ConnectContactResultsToDrakeVisualizer(builder, plant, scene_graph)

# Finailze the diagram
diagram = builder.Build()
diagram_context = diagram.CreateDefaultContext()
plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)

#####################################
## Solve Trajectory Optimization
#####################################
#
## Create a system model (w/o visualizer) to do the optimization over
#builder_ = DiagramBuilder()
#plant_, scene_graph_ = AddMultibodyPlantSceneGraph(builder_, dt)
#plant_ = create_system_model(plant_)
#builder_.ExportInput(plant_.get_actuation_input_port(), "control")
#system_ = builder_.Build()
#
## Set up the optimizer
#num_steps = int(T/dt)
#ilqr = IterativeLinearQuadraticRegulator(system_, num_steps, 
#        beta=0.9, delta=1e-2, gamma=0)
#
## Define the optimization problem
#ilqr.SetInitialState(x0)
#ilqr.SetTargetState(x_nom)
#ilqr.SetRunningCost(dt*Q, dt*R)
#ilqr.SetTerminalCost(Qf)
#
## Set initial guess
#plant.SetPositionsAndVelocities(plant_context, x0)
#tau_g = -plant.CalcGravityGeneralizedForces(plant_context)
#S = plant.MakeActuationMatrix().T
#u_gravity_comp = S@np.repeat(tau_g[np.newaxis].T, num_steps-1, axis=1)
#
##u_guess = np.zeros((plant.num_actuators(),num_steps-1))
#u_guess = 0.9*u_gravity_comp
#ilqr.SetInitialGuess(u_guess)
#
## Solve the optimization problem
#states, inputs, solve_time, optimal_cost = ilqr.Solve()
#print(f"Solved in {solve_time} seconds using iLQR")
#print(f"Optimal cost: {optimal_cost}")
#timesteps = np.arange(0.0,T,dt)
#
######################################
## Playback
######################################
#
#while True:
#    plant.get_actuation_input_port().FixValue(plant_context, 
#            np.zeros(plant.num_actuators()))
#    # Just keep playing back the trajectory
#    for i in range(len(timesteps)):
#        t = timesteps[i]
#        x = states[:,i]
#
#        diagram_context.SetTime(t)
#        plant.SetPositionsAndVelocities(plant_context, x)
#        diagram.Publish(diagram_context)
#
#        time.sleep(1/playback_rate*dt-4e-4)
#    time.sleep(1)

####################################
# Run Simulation
####################################

# Fix zero input for now
plant.get_actuation_input_port().FixValue(plant_context, np.zeros(plant.num_actuators()))

# Set initial state
plant.SetPositionsAndVelocities(plant_context, x0)

# Simulate the system
simulator = Simulator(diagram, diagram_context)
simulator.set_target_realtime_rate(playback_rate)
simulator.set_publish_every_time_step(True)

simulator.AdvanceTo(T)
