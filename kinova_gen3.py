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

T = 0.4
dt = 1e-2
playback_rate = 0.2

# Some useful joint angle definitions
q_home = np.array([0, np.pi/12, np.pi, 4.014-2*np.pi, 0, 0.9599, np.pi/2])
q_retract = np.array([0, 5.93-2*np.pi, np.pi, 3.734-2*np.pi, 0, 5.408-2*np.pi, np.pi/2])

# Initial state
x0 = np.hstack([q_retract, np.zeros(7)])

# Target state
x_nom = np.hstack([q_home, np.zeros(7)])

# Quadratic cost
Q = 5*np.eye(14)
R = 0.1*np.eye(7)
Qf = 10*np.eye(14)

# Contact model parameters
dissipation = 1.0              # controls "bounciness" of collisions: lower is bouncier
hydroelastic_modulus = 2e7     # controls "squishiness" of collisions: lower is squishier
resolution_hint = 0.05         # smaller means a finer mesh

contact_model = ContactModel.kHydroelastic  # Hydroelastic, Point, or HydroelasticWithFallback
mesh_type = HydroelasticContactRepresentation.kPolygon  # Triangle or Polygon

####################################
# Tools for system setup
####################################

def create_system_model(plant):

    # Add the kinova arm model from urdf
    # (rigid hydroelastic contact included)
    urdf = "models/kinova_gen3/urdf/GEN3_URDF_V12.urdf"
    arm = Parser(plant).AddModelFromFile(urdf)
    plant.WeldFrames(plant.world_frame(),
                     plant.GetFrameByName("base_link", arm))

    # Add an unactuated gripper from urdf
    # (rigid hydroelastic contact included)

    # Add a ground with compliant hydroelastic contact
    ground_props = ProximityProperties()
    AddCompliantHydroelasticPropertiesForHalfSpace(1.0,hydroelastic_modulus,ground_props)
    AddContactMaterial(dissipation=dissipation, friction=CoulombFriction(), properties=ground_props)
    plant.RegisterCollisionGeometry(plant.world_body(), RigidTransform(),
            HalfSpace(), "ground_collision", ground_props)

    # Add a ball with compliant hydroelastic contact

    ## Add a ball with compliant hydroelastic contact to the end of the cart-pole system
    #radius = 0.05
    #pole = plant.GetBodyByName("Pole")
    #X_BP = RigidTransform()
    #ball_props = ProximityProperties()
    #AddCompliantHydroelasticProperties(resolution_hint, hydroelastic_modulus, ball_props)
    #if contact_model == ContactModel.kPoint:
    #    AddContactMaterial(friction=CoulombFriction(), properties=ball_props)
    #else:
    #    AddContactMaterial(dissipation=dissipation, friction=CoulombFriction(), properties=ball_props)
    #plant.RegisterCollisionGeometry(pole, X_BP, Sphere(radius), "collision", ball_props)
    #orange = np.array([1.0, 0.55, 0.0, 0.5])
    #plant.RegisterVisualGeometry(pole, X_BP, Sphere(radius), "visual", orange)
    #
    ## Add a wall with rigid hydroelastic contact
    #l,w,h = (0.1,1,2)   
    #I_W = SpatialInertia(1, np.zeros(3), UnitInertia.SolidBox(l,w,h))
    #wall_instance = plant.AddModelInstance("wall")
    #wall = plant.AddRigidBody("wall", wall_instance, I_W)
    #wall_frame = plant.GetFrameByName("wall", wall_instance)
    #X_W = RigidTransform()
    #X_W.set_translation([-0.5,0,0])
    #plant.WeldFrames(plant.world_frame(), wall_frame, X_W)
    #
    #plant.RegisterVisualGeometry(wall, RigidTransform(), Box(l,w,h), "wall_visual", orange)
    #
    #wall_props = ProximityProperties()
    #AddRigidHydroelasticProperties(wall_props)
    #if contact_model == ContactModel.kPoint:
    #    AddContactMaterial(friction=CoulombFriction(), properties=wall_props)
    #else:
    #    AddContactMaterial(dissipation=dissipation, friction=CoulombFriction(), properties=wall_props)
    #plant.RegisterCollisionGeometry(wall, RigidTransform(), 
    #        Box(l,w,h), "wall_collision", wall_props)
    
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
ilqr = IterativeLinearQuadraticRegulator(plant_, plant_context_, num_steps, beta=0.5)

# Define the optimization problem
ilqr.SetInitialState(x0)
ilqr.SetTargetState(x_nom)
ilqr.SetRunningCost(dt*Q, dt*R)
ilqr.SetTerminalCost(Qf)

# Set initial guess (gravity compensation)
plant.SetPositionsAndVelocities(plant_context, x0)
tau_g = -plant.CalcGravityGeneralizedForces(plant_context)

u_guess = np.repeat(tau_g[np.newaxis].T, num_steps-1, axis=1)
ilqr.SetInitialGuess(u_guess)

# Solve the optimization problem
states, inputs, solve_time, optimal_cost = ilqr.Solve()
print(f"Solved in {solve_time} seconds using iLQR")
print(f"Optimal cost: {optimal_cost}")
timesteps = np.arange(0.0,T,dt)

######################################
## Playback
######################################

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

#####################################
## Run Simulation
#####################################

## Fix zero input for now
#plant.get_actuation_input_port().FixValue(plant_context, np.zeros(plant.num_actuators()))
#
## Set initial state
#plant.SetPositionsAndVelocities(plant_context, x0)
#
## Simulate the system
#simulator = Simulator(diagram, diagram_context)
#simulator.set_target_realtime_rate(playback_rate)
#simulator.set_publish_every_time_step(True)
#
#simulator.AdvanceTo(T)
