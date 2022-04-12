#!/usr/bin/env python

##
#
# Swing-up control of an acrobot
#
##

import numpy as np
from pydrake.all import *
from ilqr import IterativeLinearQuadraticRegulator
import time

# Choose what to do
optimize = True
playback = True
debug_gradients = False

####################################
# Parameters
####################################

T = 1.5        # total simulation time (S)
dt = 1e-2      # simulation timestep

# Solver method
# must be "ilqr" or "sqp"
method = "ilqr"
MPC = False      # MPC only works with ilqr for now

# Initial state
x0 = np.array([0,0,0,0])
#x0 = np.array([-0.22193181,  0.44017773, -5.15604732, 10.3126347 ])

# Target state
x_nom = np.array([np.pi,0,0,0])

# Quadratic cost int_{0^T} (x'Qx + u'Ru) + x_T*Qf*x_T
Q = 0.01*np.diag([0,0,1,1])
R = 0.01*np.eye(1)
Qf = 100*np.diag([1,1,1,1])

####################################
# Tools for system setup
####################################

def create_system_model(plant):
    urdf = FindResourceOrThrow("drake/examples/acrobot/Acrobot.urdf")
    robot = Parser(plant=plant).AddModelFromFile(urdf)
    plant.Finalize()
    plant.set_name("plant")
    return plant

####################################
# Create system diagram
####################################

builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, dt)
plant = create_system_model(plant)
assert plant.geometry_source_is_registered()

controller = builder.AddSystem(ConstantVectorSource(np.zeros(1)))
builder.Connect(
        controller.get_output_port(),
        plant.get_actuation_input_port())

DrakeVisualizer().AddToBuilder(builder, scene_graph)

diagram = builder.Build()
diagram_context = diagram.CreateDefaultContext()
plant_context = diagram.GetMutableSubsystemContext(
        plant, diagram_context)

#####################################
# Solve Trajectory Optimization
#####################################

if optimize:
    # System model for the trajectory optimizer
    builder_ = DiagramBuilder()
    plant_ = builder_.AddSystem(MultibodyPlant(dt))
    plant_ = create_system_model(plant_)
    builder_.ExportInput(plant_.get_actuation_input_port(), "control")
    system_ = builder_.Build()

    #-----------------------------------------
    # DDP method
    #-----------------------------------------

    def solve_ilqr(solver, x0, u_guess):
        """
        Convienience function for solving the optimization
        problem from the given initial state with the given
        guess of control inputs.
        """
        solver.SetInitialState(x0)
        solver.SetInitialGuess(u_guess)

        states, inputs, solve_time, optimal_cost = solver.Solve()
        return states, inputs, solve_time, optimal_cost

    if method == "ilqr":
        # Set up optimizer
        num_steps = int(T/dt)
        ilqr = IterativeLinearQuadraticRegulator(system_, num_steps, 
                beta=0.5, autodiff=False)

        # Define the problem
        ilqr.SetTargetState(x_nom)
        ilqr.SetRunningCost(dt*Q, dt*R)
        ilqr.SetTerminalCost(Qf)

        # Set initial guess
        u_guess = np.zeros((1,num_steps-1))

        if MPC:
            # MPC parameters
            num_resolves = 50    # total number of times to resolve the optimizaiton problem
            replan_steps = 2     # number of timesteps after which to move the horizon and
                                 # re-solve the MPC problem (>0)
            total_num_steps = num_steps + replan_steps*num_resolves
            total_T = total_num_steps*dt

            states = np.zeros((4,total_num_steps))

            # Solve to get an initial trajectory
            x, u, _, _ = solve_ilqr(ilqr, x0, u_guess)
            states[:,0:num_steps] = x

            for i in range(num_resolves):
                # Set new state and control input
                last_u = u[:,-1]
                u_guess = np.block([
                    u[:,replan_steps:],    # keep same control inputs from last optimal sol'n
                    np.repeat(last_u[np.newaxis].T,replan_steps,axis=1)  # for new timesteps copy
                    ])                                                   # the last known control input
                x0 = x[:,replan_steps]

                # Resolve the optimization
                x, u, _, _ = solve_ilqr(ilqr, x0, u_guess)

                # Save the result for playback
                start_idx = (i+1)*replan_steps
                end_idx = start_idx + num_steps
                states[:,start_idx:end_idx] = x

            timesteps = np.arange(0.0,total_T,dt)
        else:
            states, inputs, solve_time, optimal_cost = solve_ilqr(ilqr, x0, u_guess)
            print(f"Solved in {solve_time} seconds using iLQR")
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

if playback:
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

###################################
# Do some gradient debugging
###################################

if debug_gradients:
    import matplotlib.pyplot as plt

    u0 = np.zeros(plant.num_actuators())
    nx = plant.num_multibody_states()
    nq = plant.num_positions()
    nv = plant.num_velocities()

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
    fx_ad = G[:,:nx]
    fu_ad = G[:,nx:]

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
    dq_dq_err = np.amax(np.abs(err[:nq,:nq]))
    dv_dq_err = np.amax(np.abs(err[nq:,:nq]))
    dv_dv_err = np.amax(np.abs(err[nq:,nq:]))
    dq_dv_err = np.amax(np.abs(err[:nq,nq:]))

    print("\nMax errors:")
    print(f"    dq_dq {dq_dq_err}")
    print(f"    dq_dv {dq_dv_err}")
    print(f"    dv_dq {dv_dq_err}")
    print(f"    dv_dv {dv_dv_err}")

    plt.figure()
    plt.imshow(err)
    plt.title("fx error")

    plt.imshow(fx_ad, vmin=min_, vmax=max_)

    plt.show()

