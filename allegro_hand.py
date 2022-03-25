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

# Choose what to do
simulate = False
optimize = True 
playback = True
debug_gradients = False

####################################
# Parameters
####################################

T = 0.1               # total simulation time (S)
dt = 1e-3             # simulation timestep
playback_rate = 1.0   # simulation rate

# Initial state (hand has 16 acutated DoFs)
q0 = np.zeros(16)
v0 = np.zeros(16)
x0 = np.hstack([q0,v0])

# Target state
x_nom = 0.5 + np.zeros(32)

# Quadratic cost int_{0^T} (x'Qx + u'Ru) + x_T*Qf*x_T
Qq_hand = 0.1*np.ones(16)
Qv_hand = 0.01*np.ones(16)

Q = np.diag(np.hstack([Qq_hand, Qv_hand]))
R = np.eye(16)
Qf = Q

####################################
# Tools for system setup
####################################

def create_system_model(builder, dt):
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, dt)

    # Add allegro hand model
    sdf = "models/allegro_hand/allegro_hand.sdf"
    hand = Parser(plant=plant).AddModelFromFile(sdf)

    # Fix the hand position in the world
    X_hand = RigidTransform()
    X_hand.set_translation([0,0,0.1])
    X_hand.set_rotation(RotationMatrix(RollPitchYaw([0,-np.pi/2,0])))
    plant.WeldFrames(plant.world_frame(), 
                     plant.GetFrameByName("hand_root", hand),
                     X_hand)

    # Add something to manipulate
   
    # Turn off gravity
    #plant.mutable_gravity_field().set_gravity_vector([0,0,0])

    plant.Finalize()
    plant.set_name("plant")
    return plant, scene_graph

####################################
# Create system diagram
####################################
builder = DiagramBuilder()
plant, scene_graph = create_system_model(builder, dt)

# Connect to visualizer
DrakeVisualizer().AddToBuilder(builder, scene_graph)
#ConnectContactResultsToDrakeVisualizer(builder, plant, scene_graph)

# Finalize the diagram
diagram = builder.Build()
diagram_context = diagram.CreateDefaultContext()
plant_context = diagram.GetMutableSubsystemContext(
        plant, diagram_context)

#####################################
# Solve Trajectory Optimization
#####################################

if optimize:
    # Create system model for controller
    builder_ = DiagramBuilder()
    plant_, scene_graph_ = create_system_model(builder_, dt)
    builder_.ExportInput(plant_.get_actuation_input_port(), "control")
    system_ = builder_.Build()

    # Set up the optimizer
    num_steps = int(T/dt)
    ilqr = IterativeLinearQuadraticRegulator(system_, num_steps, 
            beta=0.9, autodiff=True)

    # Define initial and target states
    ilqr.SetInitialState(x0)
    ilqr.SetTargetState(x_nom)

    # Define cost function
    ilqr.SetRunningCost(dt*Q, dt*R)
    ilqr.SetTerminalCost(Qf)

    # Set initial guess
    plant.SetPositionsAndVelocities(plant_context, x0)
    tau_g = -plant.CalcGravityGeneralizedForces(plant_context)
    S = plant.MakeActuationMatrix().T
    #u_guess = np.zeros((16,num_steps-1))
    u_guess = S@np.repeat(tau_g[np.newaxis].T, num_steps-1, axis=1)
    ilqr.SetInitialGuess(u_guess)

    states, inputs, solve_time, optimal_cost = ilqr.Solve()
    print(f"Solved in {solve_time} seconds using iLQR")
    print(f"Optimal cost: {optimal_cost}")
    timesteps = np.arange(0.0,T,dt)

#####################################
# Playback
#####################################

if playback:
    
    # Fix input as zero, since it isn't used
    plant.get_actuation_input_port().FixValue(plant_context, np.zeros(plant.num_actuators()))

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

####################################
# Run Simulation
####################################

if simulate:
    # Fix zero input for now
    plant.get_actuation_input_port().FixValue(plant_context, np.zeros(plant.num_actuators()))

    # Set initial state
    plant.SetPositionsAndVelocities(plant_context, x0)

    # Simulate the system
    simulator = Simulator(diagram, diagram_context)
    simulator.set_target_realtime_rate(playback_rate)

    simulator.AdvanceTo(T)

###################################
# Do some gradient debugging
###################################

if debug_gradients:
    import matplotlib.pyplot as plt

    u0 = np.zeros(plant.num_actuators())

    # Get gradients via custom method
    plant.get_actuation_input_port().FixValue(plant_context, u0)
    plant.SetPositionsAndVelocities(plant_context, x0)
    st = time.time()
    x_next, fx, fu = plant.DiscreteDynamicsWithApproximateGradients(plant_context)
    print("our time: ", time.time()-st)

    # Get gradients via autodiff
    diagram_ad = diagram.ToAutoDiffXd()
    context_ad = diagram_ad.CreateDefaultContext()
    plant_ad = diagram_ad.GetSubsystemByName("plant")
    plant_context_ad = diagram_ad.GetMutableSubsystemContext(plant_ad, context_ad)
    xu_ad = InitializeAutoDiff(np.hstack([x0,u0]))
    x_ad = xu_ad[:plant.num_multibody_states()]
    u_ad = xu_ad[plant.num_multibody_states():]
    
    plant_ad.get_actuation_input_port().FixValue(plant_context_ad, u_ad)
    plant_ad.SetPositionsAndVelocities(plant_context_ad, x_ad)
    state = context_ad.get_discrete_state()
    st = time.time()
    diagram_ad.CalcDiscreteVariableUpdates(context_ad, state)
    print("autodiff time: ", time.time()-st)
    x_next_ad = state.get_vector().CopyToVector()
    G = ExtractGradient(x_next_ad)
    fx_ad = G[:,:plant.num_multibody_states()]
    fu_ad = G[:,plant.num_multibody_states():]

    # zoom in on a particular part
    #fx = fx[16:,:16]
    #fx_ad = fx_ad[16:,:16]

    # Visualize the gradients
    min_ = np.amin(np.hstack([fx,fx_ad]))
    max_ = np.amax(np.hstack([fx,fx_ad]))

    plt.subplot(2,2,1)
    plt.imshow(fx, vmin=min_, vmax=max_)
    plt.title("fx ours")
    plt.subplot(2,2,2)
    plt.imshow(fx_ad, vmin=min_, vmax=max_)
    plt.title("fx autodiff")
    plt.subplot(2,2,3)
    plt.imshow(fu)
    plt.title("fu ours")
    plt.subplot(2,2,4)
    plt.imshow(fu_ad)
    plt.title("fu autodiff")

    # Do some computations related to the error
    err = fx-fx_ad
    dq_dq_err = np.amax(np.abs(err[:16,:16]))
    dv_dq_err = np.amax(np.abs(err[16:,:16]))
    dv_dv_err = np.amax(np.abs(err[16:,16:]))
    dq_dv_err = np.amax(np.abs(err[:16,16:]))

    print("\nMax errors:")
    print(f"    dq_dq {dq_dq_err}")
    print(f"    dq_dv {dq_dv_err}")
    print(f"    dv_dq {dv_dq_err}")
    print(f"    dv_dv {dv_dv_err}")

    plt.figure()
    plt.imshow(err)
    plt.title("fx error")

    plt.show()

