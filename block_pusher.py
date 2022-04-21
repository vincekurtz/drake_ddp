#!/usr/bin/env python

##
#
# Use iLQR over hydroelastic contact to
# push a block around. 
#
##

import numpy as np
from pydrake.all import *
from ilqr import IterativeLinearQuadraticRegulator

# Choose what to do
simulate = True
optimize = False

####################################
# Parameters
####################################

# Simulation parameters
T = 1.0
dt = 1e-3
playback_rate = 1.0

# Contact model parameters
dissipation = 0.0              # controls "bounciness" of collisions: lower is bouncier
hydroelastic_modulus = 2e5     # controls "squishiness" of collisions: lower is squishier
resolution_hint = 0.05         # smaller means a finer mesh
penetration_allowance = 0.02   # controls "softness" of collisions for point contact model
mu = 0.5                       # friction coefficient
contact_model = ContactModel.kPoint  # Hydroelastic, Point, or HydroelasticWithFallback
mesh_type = HydroelasticContactRepresentation.kPolygon  # Triangle or Polygon

# Block parameters
mass = 0.1
length = 0.1
width = 0.1
height = 0.05

# Pusher parameters
pusher_mass = 1.0    # large inertia makes sense if pusher is connected to robot arm
pusher_radius = 0.01

# Initial state
q0_pusher = np.array([0,length/2+pusher_radius])
q0_block = np.array([1,0,0,0, 0,0,height/2])
q0 = np.hstack([q0_pusher, q0_block])
v0 = np.zeros(8)
x0 = np.hstack([q0,v0])

# Target state
x_nom = np.array([0,np.pi,0,0])

# Quadratic cost
Q = np.diag([1,1,0.1,0.1])
R = 0.001*np.eye(1)
Qf = np.diag([500,500,1,1])

####################################
# Tools for system setup
####################################

def create_system_model(builder):
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, dt)

    # Add a block with friction
    block = plant.AddModelInstance("block")
    I = SpatialInertia(mass=mass, p_PScm_E=np.array([0., 0., 0.]),
                       G_SP_E=UnitInertia.SolidBox(length, width, height))
    block_body = plant.AddRigidBody("block_body", block, I)

    plant.RegisterCollisionGeometry(block_body, RigidTransform(),
            Box(length, width, height), "box_collision", CoulombFriction(mu,mu))
    plant.RegisterVisualGeometry(block_body, RigidTransform(),
            Box(length, width, height), "box_visual", [0.5, 0.5, 0.9, 1.0])

    # Add ground with friction
    plant.RegisterCollisionGeometry(plant.world_body(), RigidTransform(),
            HalfSpace(), "ground_collision", CoulombFriction(mu,mu))

    # Add a pusher
    pusher = plant.AddModelInstance("pusher")
    dummy1 = plant.AddRigidBody("dummy1", pusher, 
            SpatialInertia(0, [0,0,0], UnitInertia(0,0,0)))
    dummy2 = plant.AddRigidBody("dummy2", pusher, 
            SpatialInertia(0, [0,0,0], UnitInertia(0,0,0)))
    pusher_body = plant.AddRigidBody("pusher_body", pusher,
            SpatialInertia(pusher_mass, [0,0,0], UnitInertia.SolidSphere(pusher_radius)))

    plant.RegisterCollisionGeometry(pusher_body, RigidTransform(),
            Sphere(pusher_radius), "pusher_collision", CoulombFriction(mu,mu))
    plant.RegisterVisualGeometry(pusher_body,
            RigidTransform(), Sphere(pusher_radius), "pusher_visual", [0.9, 0.5, 0.5, 1.0])

    pusher_x = plant.AddJoint(PrismaticJoint("pusher_x", plant.world_frame(),
        plant.GetFrameByName("dummy1"), [1,0,0], -0.5, 0.5))
    pusher_y = plant.AddJoint(PrismaticJoint("pusher_y", plant.GetFrameByName("dummy1"),
        plant.GetFrameByName("dummy2"), [0,1,0], -0.5, 0.5))
    pusher_z = plant.AddJoint(WeldJoint("pusher_z", plant.GetFrameByName("dummy2"),
        plant.GetFrameByName("pusher_body"), 
        RigidTransform(RotationMatrix(),[0,0,height/2])))

    plant.AddJointActuator("pusher_x", pusher_x)
    plant.AddJointActuator("pusher_y", pusher_y)
    pusher_y.set_default_positions([length/2+pusher_radius])

    # Turn off gravity
    #plant.mutable_gravity_field().set_gravity_vector([0,0,0])
    
    # Choose contact model
    plant.set_contact_surface_representation(mesh_type)
    plant.set_contact_model(contact_model)
    plant.Finalize()

    return plant, scene_graph

####################################
# Create system diagram
####################################
builder = DiagramBuilder()
plant, scene_graph = create_system_model(builder)

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

if optimize:
    # Create system model for the solver to use. This system model
    # has a single input port for the control and doesn't include
    # any visualizer stuff. 
    builder_ = DiagramBuilder()
    plant_, scene_graph_ = create_system_model(builder_)
    builder_.ExportInput(plant_.get_actuation_input_port(), "control")
    system_ = builder_.Build()

    # Set up the optimizer
    num_steps = int(T/dt)
    ilqr = IterativeLinearQuadraticRegulator(system_, num_steps, beta=0.5, autodiff=True)

    # Define the optimization problem
    ilqr.SetInitialState(x0)
    ilqr.SetTargetState(x_nom)
    ilqr.SetRunningCost(dt*Q, dt*R)
    ilqr.SetTerminalCost(Qf)

    # Set initial guess
    #np.random.seed(0)
    #u_guess = 0.01*np.random.normal(size=(1,num_steps-1))
    u_guess = np.zeros((1,num_steps-1))
    ilqr.SetInitialGuess(u_guess)

    # Solve the optimization problem
    states, inputs, solve_time, optimal_cost = ilqr.Solve()
    print(f"Solved in {solve_time} seconds using iLQR")
    print(f"Optimal cost: {optimal_cost}")
    timesteps = np.arange(0.0,T,dt)

    # Set initial conditions
    plant.get_actuation_input_port().FixValue(plant_context, 0)
    plant.SetPositionsAndVelocities(plant_context, x0)

#####################################
# Playback
#####################################

    while True:
        plant.get_actuation_input_port().FixValue(plant_context, 0)
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
    u = np.zeros(plant.num_actuators())
    u[1] -= 1.0
    plant.get_actuation_input_port().FixValue(plant_context, u)

    # Set initial state
    plant.SetPositionsAndVelocities(plant_context, x0)

    # Simulate the system
    simulator = Simulator(diagram, diagram_context)
    simulator.set_target_realtime_rate(playback_rate)

    simulator.AdvanceTo(T)
