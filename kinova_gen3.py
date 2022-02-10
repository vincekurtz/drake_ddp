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

T = 0.5
dt = 1e-2
playback_rate = 0.2

# Some useful joint angle definitions
q_home = np.array([0, np.pi/12, np.pi, 4.014-2*np.pi, 0, 0.9599, np.pi/2])
q_retract = np.array([0, 5.93-2*np.pi, np.pi, 3.734-2*np.pi, 0, 5.408-2*np.pi, np.pi/2])
q_start = np.array([0.0, np.pi/4+0.15, np.pi, 4.4-2*np.pi, 0, 1.2, np.pi/2])

q_ball_start = np.array([0,0,0,1,0.5,0,0.1])
q_ball_target = np.array([0,0,0,1,0.6,0.0,0.1])

# Initial state
x0 = np.hstack([q_start, q_ball_start, np.zeros(13)])

# Target state
x_nom = np.hstack([q_start, q_ball_target, np.zeros(13)])

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
dissipation = 1.0              # controls "bounciness" of collisions: lower is bouncier
hydroelastic_modulus = 2e6     # controls "squishiness" of collisions: lower is squishier
resolution_hint = 0.05         # smaller means a finer mesh

contact_model = ContactModel.kHydroelasticWithFallback  # Hydroelastic, Point, or HydroelasticWithFallback
mesh_type = HydroelasticContactRepresentation.kTriangle  # Triangle or Polygon

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
    #AddContactMaterial(dissipation=dissipation, friction=CoulombFriction(), properties=ground_props)
    AddContactMaterial(friction=CoulombFriction(), properties=ground_props)
    plant.RegisterCollisionGeometry(plant.world_body(), RigidTransform(),
            HalfSpace(), "ground_collision", ground_props)

    # Add a ball with compliant hydroelastic contact
    radius = 0.1
    mass = 0.1
    I = SpatialInertia(mass, np.zeros(3), UnitInertia.SolidSphere(radius))
    ball_instance = plant.AddModelInstance("ball")
    ball = plant.AddRigidBody("ball",ball_instance, I)
    X_ball = RigidTransform()

    ball_props = ProximityProperties()
    AddCompliantHydroelasticProperties(resolution_hint, hydroelastic_modulus,ball_props)
    #AddContactMaterial(dissipation=dissipation, friction=CoulombFriction(), properties=ball_props)
    AddContactMaterial(friction=CoulombFriction(), properties=ball_props)
    plant.RegisterCollisionGeometry(ball, X_ball, Sphere(radius),
            "ball_collision", ball_props)

    color = np.array([1.0,0.55,0.0, 0.5])
    plant.RegisterVisualGeometry(ball, X_ball, Sphere(radius), "ball_visual", color)
    
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
builder_.ExportInput(plant_.get_actuation_input_port(), "control")
system_ = builder_.Build()

# Set up the optimizer
num_steps = int(T/dt)
ilqr = IterativeLinearQuadraticRegulator(system_, num_steps, 
        beta=0.9, delta=1e-2, gamma=0)

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
u_guess = 0.9*u_gravity_comp
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
#
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
