#!/usr/bin/env python

##
#
# Simple simulation to test Drake's new hydroelastic 
# contact model.
#
##

import numpy as np
from pydrake.all import *
from ilqr import IterativeLinearQuadraticRegulator


####################################
# Parameters
####################################

T = 1.0 
dt = 1e-2

# Initial state
x0 = np.array([0,np.pi+0.2,-0.2,0])

# Target state
x_nom = np.array([0,np.pi,0,0])

# Quadratic cost
Q = np.diag([5,5,0.01,0.01])
R = 0.01*np.eye(1)
Qf = np.diag([100,100,10,10])

# Contact model parameters
dissipation = 0.0              # controls "bounciness" of collisions: lower is bouncier
hydroelastic_modulus = 2e5     # controls "squishiness" of collisions: lower is squishier
resolution_hint = 0.05         # smaller means a finer mesh

contact_model = ContactModel.kHydroelastic  # Hydroelastic, Point, or HydroelasticWithFallback
mesh_type = HydroelasticContactRepresentation.kPolygon  # Triangle or Polygon

####################################
# Tools for system setup
####################################

def create_system_model(plant):
    # Add the cart-pole system
    sdf = FindResourceOrThrow("drake/examples/multibody/cart_pole/cart_pole.sdf")
    robot = Parser(plant=plant).AddModelFromFile(sdf)
    
    # Add a ball with compliant hydroelastic contact to the end of the cart-pole system
    radius = 0.05
    pole = plant.GetBodyByName("Pole")
    X_BP = RigidTransform()
    ball_props = ProximityProperties()
    AddCompliantHydroelasticProperties(resolution_hint, hydroelastic_modulus, ball_props)
    #AddContactMaterial(dissipation=dissipation, friction=CoulombFriction(), properties=ball_props)
    AddContactMaterial(friction=CoulombFriction(), properties=ball_props)
    plant.RegisterCollisionGeometry(pole, X_BP, Sphere(radius), "collision", ball_props)
    orange = np.array([1.0, 0.55, 0.0, 0.5])
    plant.RegisterVisualGeometry(pole, X_BP, Sphere(radius), "visual", orange)
    
    # Add a wall with rigid hydroelastic contact
    l,w,h = (0.1,1,2)   
    I_W = SpatialInertia(1, np.zeros(3), UnitInertia.SolidBox(l,w,h))
    wall_instance = plant.AddModelInstance("wall")
    wall = plant.AddRigidBody("wall", wall_instance, I_W)
    wall_frame = plant.GetFrameByName("wall", wall_instance)
    X_W = RigidTransform()
    X_W.set_translation([-0.5,0,0])
    plant.WeldFrames(plant.world_frame(), wall_frame, X_W)
    
    plant.RegisterVisualGeometry(wall, RigidTransform(), Box(l,w,h), "wall_visual", orange)
    
    wall_props = ProximityProperties()
    AddRigidHydroelasticProperties(wall_props)
    #AddContactMaterial(dissipation=dissipation, friction=CoulombFriction(), properties=wall_props)
    AddContactMaterial(friction=CoulombFriction(), properties=wall_props)
    plant.RegisterCollisionGeometry(wall, RigidTransform(), 
            Box(l,w,h), "wall_collision", wall_props)
    
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

####################################
# Solve Trajectory Optimization
####################################

# Create a system model (w/o visualizer) to do the optimization over
builder_ = DiagramBuilder()
plant_, scene_graph_ = AddMultibodyPlantSceneGraph(builder_, dt)
plant_ = create_system_model(plant_)
diagram_ = builder_.Build()

# Convert this system model to AutoDiff type
diagram_ = diagram_.ToAutoDiffXd()
diagram_context_ = diagram_.CreateDefaultContext()
plant_ = diagram_.GetSubsystemByName("plant")
plant_context_ = diagram_.GetMutableSubsystemContext(plant_, diagram_context_)

# Set up the optimizer
num_steps = int(T/dt)
ilqr = IterativeLinearQuadraticRegulator(plant_, plant_context_, num_steps, beta=0.9)

# Define the optimization problem
ilqr.SetInitialState(x0)
ilqr.SetTargetState(x_nom)
ilqr.SetRunningCost(dt*Q, dt*R)
ilqr.SetTerminalCost(Qf)

# Set initial guess
u_guess = np.zeros((1,num_steps-1))
ilqr.SetInitialGuess(u_guess)

# Solve the optimization problem
states, inputs, solve_time, optimal_cost = ilqr.Solve()
print(f"Solved in {solve_time} seconds using iLQR")
print(f"Optimal cost: {optimal_cost}")
timesteps = np.arange(0.0,T,dt)

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

#####################################
## Run Simulation
#####################################
#
## Fix zero input for now
#plant.get_actuation_input_port().FixValue(plant_context, 0)
#
## Set initial state
#plant.SetPositionsAndVelocities(plant_context, x0)
#
## Simulate the system
#simulator = Simulator(diagram, diagram_context)
#simulator.set_target_realtime_rate(1.0)
#
#simulator.AdvanceTo(T)
