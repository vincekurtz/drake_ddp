#!/usr/bin/env python

##
#
# Contact-implicit trajectory optimization for
# automatic gait discovery on a Mini Cheetah quadruped.
#
##

import numpy as np
from pydrake.all import *
from ilqr import IterativeLinearQuadraticRegulator

####################################
# Parameters
####################################

T = 0.2
dt = 5e-3
playback_rate = 0.2
target_vel = 0.50   # m/s

# MPC parameters
num_resolves = 20  # total number of times to resolve the optimizaiton problem
replan_steps = 4    # number of timesteps after which to move the horizon and
                    # re-solve the MPC problem (>0)

# Some useful definitions
q0 = np.asarray([ 1.0, 0.0, 0.0, 0.0,     # base orientation
                  0.0, 0.0, 0.29,          # base position
                  0.0,-0.8, 1.6,
                  0.0,-0.8, 1.6,
                  0.0,-0.8, 1.6,
                  0.0,-0.8, 1.6])
u_stand = np.array([ 0.16370625,  0.42056475, -3.06492254,  0.16861717,  0.14882384,
       -2.43250739,  0.08305763,  0.26016952, -2.74586461,  0.08721941,
        0.02331732, -2.18319231])

# Initial state
x0 = np.hstack([q0, np.zeros(18)])

# Target state
x_nom = np.hstack([q0, np.zeros(18)])
x_nom[4] += target_vel*T  # base x position
x_nom[22] += target_vel  # base x velocity

# Quadratic cost
Qq_base = np.ones(7)
Qq_base[0:4] += 2
Qv_base = np.ones(6)

Qq_legs = 0.0*np.ones(12)
Qv_legs = 0.01*np.ones(12)

Q = np.diag(np.hstack([Qq_base,Qq_legs,0.01*Qv_base,Qv_legs]))
R = 0.01*np.eye(12)
Qf = np.diag(np.hstack([5*Qq_base,0.1+Qq_legs,Qv_base,Qv_legs]))

# Contact model parameters
contact_model = ContactModel.kHydroelastic  # Hydroelastic, Point, or HydroelasticWithFallback
mesh_type = HydroelasticContactRepresentation.kPolygon  # Triangle or Polygon

mu = 0.5

dissipation = 0
hydroelastic_modulus = 5e6
resolution_hint = 0.1

####################################
# Tools for system setup
####################################

def create_system_model(plant):

    # Add the kinova arm model from urdf (rigid hydroelastic contact included)
    urdf = "models/mini_cheetah/mini_cheetah_mesh.urdf"
    arm = Parser(plant).AddModelFromFile(urdf)

    # Add a ground with compliant hydroelastic contact
    ground_props = ProximityProperties()
    AddCompliantHydroelasticProperties(resolution_hint, hydroelastic_modulus,ground_props)
    friction = CoulombFriction(mu, mu)
    AddContactMaterial(dissipation=dissipation, friction=friction, properties=ground_props)
    X_ground = RigidTransform()
    X_ground.set_translation([0,0,-0.5])
    ground_shape = Box(25,25,1)
    plant.RegisterCollisionGeometry(plant.world_body(), X_ground,
            ground_shape, "ground_collision", ground_props)
    #plant.RegisterVisualGeometry(plant.world_body(), X_ground,
    #        ground_shape, "ground_visual", np.array([1,0,0,0.5]))

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
# Solve Trajectory Optimization 
#################################### 

# Create a system model (w/o visualizer) to do the optimization over 
builder_ = DiagramBuilder() 
plant_, scene_graph_ = AddMultibodyPlantSceneGraph(builder_, dt)
plant_ = create_system_model(plant_)
builder_.ExportInput(plant_.get_actuation_input_port(), "control")
system_ = builder_.Build()

# Helper function for solving the optimization problem from the
# given initial state with the given guess of control inputs
def solve_ilqr(solver, x0, u_guess, move_target=False):
    solver.SetInitialState(x0)
    solver.SetInitialGuess(u_guess)

    if move_target:
        # update target state consistent with desired
        # velocity
        delta_t = dt*replan_steps
        x_nom[4] += target_vel*delta_t
        solver.SetTargetState(x_nom)

    states, inputs, solve_time, optimal_cost = solver.Solve()
    return states, inputs, solve_time, optimal_cost

# Set up the optimizer
num_steps = int(T/dt)
ilqr = IterativeLinearQuadraticRegulator(system_, num_steps, 
        beta=0.5, delta=1e-2, gamma=0, autodiff=False)

# Define the optimization problem
ilqr.SetTargetState(x_nom)
ilqr.SetRunningCost(dt*Q, dt*R)
ilqr.SetTerminalCost(Qf)

# Set initial guess
u_guess = np.repeat(u_stand[np.newaxis].T,num_steps-1,axis=1)

# MPC setup
total_num_steps = num_steps + replan_steps*num_resolves
total_T = total_num_steps*dt
states = np.zeros((plant.num_multibody_states(),total_num_steps))

# Solve to get an initial trajectory
st = time.time()
x, u, _, _ = solve_ilqr(ilqr, x0, u_guess)
states[:,0:num_steps] = x

# Perform additional resolves in MPC-fashion
for i in range(num_resolves):
    print(f"\nRunning resolve {i+1}/{num_resolves}\n")
    # Set new state and control input
    last_u = u[:,-1]
    u_guess = np.block([
        u[:,replan_steps:],    # keep same control inputs from last optimal sol'n
        np.repeat(last_u[np.newaxis].T,replan_steps,axis=1)  # for new timesteps copy
        ])                                                   # the last known control input
    x0 = x[:,replan_steps]

    # Resolve the optimization
    x, u, _, _ = solve_ilqr(ilqr, x0, u_guess, move_target=True)

    # Save the result for playback
    start_idx = (i+1)*replan_steps
    end_idx = start_idx + num_steps
    states[:,start_idx:end_idx] = x

    # Update the visualizer so we have a general sense of what
    # the optimizer is doing
    diagram_context.SetTime(end_idx*dt)
    plant.SetPositionsAndVelocities(plant_context, x[:,-1])
    diagram.Publish(diagram_context)
    

solve_time = time.time() - st
print(f"Solved in {solve_time} seconds using iLQR")
timesteps = np.arange(0.0,total_T,dt)

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
## Fix input
#plant.get_actuation_input_port().FixValue(plant_context, np.zeros(plant.num_actuators()))
##plant.get_actuation_input_port().FixValue(plant_context, u_stand)
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
