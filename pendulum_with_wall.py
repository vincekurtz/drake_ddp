#!/usr/bin/env python

##
#
# Swing-up control of a simple inverted pendulum
#
##

import numpy as np
from pydrake.all import *
from ilqr import IterativeLinearQuadraticRegulator
from mcilqr import MonteCarloIterativeLQR
import time
import matplotlib.pyplot as plt

####################################
# Parameters
####################################

T = 1.0         # total simulation time (S)
dt = 1e-2       # simulation timestep

# Initial state
x0 = np.array([0.85*np.pi,0])

# Target state
x_nom = np.array([np.pi,0])

# Quadratic cost int_{0^T} (x'Qx + u'Ru) + x_T*Qf*x_T
Q = 0.1*np.diag([1,1])
R = 0.01*np.eye(1)
Qf = 100*np.diag([1,1])

####################################
# Tools for system setup
####################################

def create_system_model(builder):
    """
    Create and return a plant and scene graph for the simple pendulum system.
    """
    # System parameters
    mass = 1.0
    radius = 0.01
    length = 0.5
    damping = 0.1

    # Create the plant and scene graph
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, dt)

    # Create the pendulum object
    pendulum = plant.AddModelInstance("pendulum")
    rod = plant.AddRigidBody("rod", pendulum, SpatialInertia(mass, [0,0,0],
                UnitInertia.SolidCylinder(radius, length)))
    rod_com_frame = plant.GetFrameByName("rod")
    X_base_com = RigidTransform(RotationMatrix(),[0,0,length/2])
    rod_base_frame = plant.AddFrame(FixedOffsetFrame("rod_base",
        rod_com_frame, X_base_com))
    base_joint = plant.AddJoint(RevoluteJoint("base_joint", plant.world_frame(),
        rod_base_frame, [1,0,0], damping))
    plant.AddJointActuator("base_actuator", base_joint)
    rod_shape = Cylinder(radius, length)
    rod_props = ProximityProperties()
    AddCompliantHydroelasticProperties(radius, 5e6, rod_props)
    AddContactMaterial(friction=CoulombFriction(0.5,0.5), properties=rod_props)
    plant.RegisterCollisionGeometry(rod, RigidTransform(), rod_shape,
            "rod_collision", rod_props)
    plant.RegisterVisualGeometry(rod, RigidTransform(), rod_shape, "rod_visual", 
            [0.5,0.5,0.9,1.0])

    # Add a sphere on the end of the pendulum
    ball_shape = Sphere(2*radius)
    ball = plant.AddRigidBody("ball", pendulum, SpatialInertia(mass, [0,0,0],
                UnitInertia.SolidSphere(2*radius)))
    X_ball = RigidTransform(RotationMatrix(),[0,0,-length/2])
    plant.WeldFrames(plant.GetFrameByName("rod"), plant.GetFrameByName("ball"),
            X_ball)
    ball_props = ProximityProperties()
    AddCompliantHydroelasticProperties(2*radius, 5e6, ball_props)
    AddContactMaterial(friction=CoulombFriction(0.5,0.5), properties=ball_props)
    plant.RegisterCollisionGeometry(ball, RigidTransform(), ball_shape,
            "ball_collision", ball_props)
    plant.RegisterVisualGeometry(ball, RigidTransform(), ball_shape,
            "ball_visual", [0.5,0.5,0.9,1.0])

    # Add a box to collide with
    box = plant.AddModelInstance("box")
    box_body = plant.AddRigidBody("box", box, 
            SpatialInertia(1.0, [0,0,0], UnitInertia.SolidBox(1,1,1)))
    X_box = RigidTransform(RotationMatrix(),[0,0.4,0])
    box_frame = plant.GetFrameByName("box")
    plant.WeldFrames(plant.world_frame(), box_frame, X_box)
    box_shape = Box(0.5,0.1,1.0)
    box_props = ProximityProperties()
    AddCompliantHydroelasticProperties(1.0, 5e6, box_props)
    AddContactMaterial(dissipation=1, friction=CoulombFriction(0.5,0.5), properties=box_props)
    plant.RegisterCollisionGeometry(box_body, RigidTransform(), box_shape,
            "box_collision", box_props)
    plant.RegisterVisualGeometry(box_body, RigidTransform(), box_shape, 
            "box_visual", [0.5,0.9,0.5,1.0])

    plant.Finalize()

    return plant, scene_graph

####################################
# Create system diagram
####################################

builder = DiagramBuilder()
plant, scene_graph = create_system_model(builder)
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
plant_, scene_graph_ = create_system_model(builder_)
builder_.ExportInput(plant_.get_actuation_input_port(),"control")
system_ = builder_.Build()

mc = True  # use monte-carlo/stochastic version
ns = 4     # number of monte-carlo samples to use

# Set up the optimizer
num_steps = int(T/dt)
if mc:
    ilqr = MonteCarloIterativeLQR(system_, num_steps, ns, seed=0)
else:
    ilqr = IterativeLinearQuadraticRegulator(system_, num_steps)

# Define initial and target states
if mc:
    mu = x0
    Sigma = np.diag([0.05,0.0])
    ilqr.SetInitialDistribution(mu,Sigma)
else:
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

#####################################
# Playback
#####################################

# Make some plot of the state trajectories
plt.subplot(3,1,1)
thetas = states[0::2,:]
theta_dots = states[1::2,:]
plt.plot(timesteps,thetas.T)
plt.ylabel("theta")
plt.subplot(3,1,2)
plt.plot(timesteps,theta_dots.T)
plt.ylabel("theta_dot")
plt.subplot(3,1,3)
plt.plot(timesteps[:-1],inputs.T)
plt.ylabel("torque")
plt.xlabel("time")
plt.show(block=False)
plt.pause(0.01)

j = 0
while True:

    if mc:
        # Choose which copy of the trajectory to play back
        j = (j+1) % ns
    
    # Just keep playing back the trajectory
    for i in range(len(timesteps)):
        t = timesteps[i]
        x = states[j*2:(j+1)*2,i]

        diagram_context.SetTime(t)
        plant.SetPositionsAndVelocities(plant_context, x)
        diagram.Publish(diagram_context)

        time.sleep(dt-3e-4)
    time.sleep(1)
