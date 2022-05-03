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

def set_up_visualizer_system():
    """
    Set up a system model for use with the Drake visualizer.
    """
    builder = DiagramBuilder()
    plant, scene_graph = create_system_model(builder)
    assert plant.geometry_source_is_registered()

    DrakeVisualizer().AddToBuilder(builder, scene_graph)
    ConnectContactResultsToDrakeVisualizer(builder, plant, scene_graph)

    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(
            plant, diagram_context)

    return diagram, diagram_context, plant, plant_context

def simulate_open_loop(x0, u):
    """
    Apply the given control sequence to the system from the
    given initial conditions. Shows the result on the visualizer.
    """
    N = u.shape[1]
    diagram, diagram_context, plant, plant_context = set_up_visualizer_system()

    diagram_context.SetTime(0)
    diagram_context.SetDiscreteState(x0)
    plant.get_actuation_input_port().FixValue(plant_context, [0])
    diagram.Publish(diagram_context)

    t = 0
    for i in range(N):
        diagram_context.SetTime(t)
        plant.get_actuation_input_port().FixValue(plant_context, u[:,i])

        state = diagram_context.get_mutable_discrete_state()
        diagram.CalcDiscreteVariableUpdates(diagram_context, state)
        diagram.Publish(diagram_context)

        t += dt
        time.sleep(dt-3e-4)

def playback_state_trajectories(states):
    """
    Play back the given set of state trajectories on the visualizer.
    """
    diagram, diagram_context, plant, plant_context = set_up_visualizer_system()
    plant.get_actuation_input_port().FixValue(plant_context, [0])
    ns = int(states.shape[0] / 2)  # number of state samples

    j = 0
    while True:
        # Choose which copy of the trajectory to play back
        j = (j+1) % ns
        
        # Just keep playing back the trajectory
        for i in range(len(timesteps)):
            t = timesteps[i]
            x = states[j*2:(j+1)*2,i]

            diagram_context.SetTime(t)
            diagram_context.SetDiscreteState(x)
            diagram.Publish(diagram_context)

            if i==0:
                # Wait a bit on first timestep to show initial state
                time.sleep(0.5)
            time.sleep(dt-3e-4)
        time.sleep(1)

def solve_mc_ilqr(mu0, Sigma0, x_nom, Q, R, ns):
    """
    Solve the monte-carlo iLQR problem with the given
    initial state distribution, target state, cost matrices, and 
    number of samples. 
    """
    # Create system model for the solver to use
    builder_ = DiagramBuilder()
    plant_, scene_graph_ = create_system_model(builder_)
    builder_.ExportInput(plant_.get_actuation_input_port(),"control")
    system_ = builder_.Build()

    # Set up the optimizer
    num_steps = int(T/dt)
    ilqr = MonteCarloIterativeLQR(system_, num_steps, ns, seed=0)

    # Define initial and target states
    ilqr.SetInitialDistribution(mu0,Sigma0)
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

    return states, inputs, timesteps

def plot_state_and_control(states, inputs, timesteps, block=True):
    """
    Make a quick plot of the (sampled) state and input trajectories.
    """
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
    plt.show(block=block)
    plt.pause(0.01)

if __name__=="__main__":
    # Set parameters
    T = 1.0         # total simulation time (S)
    dt = 1e-2       # simulation timestep

    x0 = np.array([0.8*np.pi,0])  # initial state
    x_nom = np.array([np.pi,0])   # target state

    Q = np.diag([100,10])   # Quadratic cost
    R = 0.01*np.eye(1)      # int_{0^T} (x'Qx + u'Ru) + x_T*Qf*x_T
    Qf = np.diag([150,10])

    # Solve the iLQR problem
    mu0 = x0
    Sigma0 = np.diag([0.05,0])
    states, inputs, timesteps = solve_mc_ilqr(mu0, Sigma0, x_nom, Q, R, ns=2)

    # Make a plot of the optimal trajectories
    plot_state_and_control(states, inputs, timesteps, block=False)

    # Play back the state trajectories in sequence
    #playback_state_trajectories(states)

    # Simulate optimal control from new initial conditions
    while True:
        x0 = np.random.multivariate_normal(mu0, Sigma0)
        simulate_open_loop(x0, inputs)
        time.sleep(1)

