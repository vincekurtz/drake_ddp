#!/usr/bin/env python

##
#
# Contact-implicit trajectory optimization for whole-arm manipulation
# using a Kinova Gen3 manipulator arm.
#
##

import time
import numpy as np
from pydrake.all import *
from ilqr import IterativeLinearQuadraticRegulator
import utils_derivs_interpolation

# Choose what to do
simulate = False   # Run a simple simulation with fixed input
optimize = True    # Find an optimal trajectory using ilqr
playback = True    # Visualize the optimal trajectory by playing it back.
                   # If optimize=False, attempts to load a previously saved
                   # trajectory from a file.

scenario = "side"   # "lift", "forward", or "side"
save_file = scenario + ".npz"

meshcat_visualisation = True

####################################
# Parameters
####################################

T = 0.5
dt = 1e-2
playback_rate = 0.125

# Some useful joint angle definitions
q_home = np.pi/180*np.array([0, 15, 180, 230, 0, 55, 90])
q_retract = np.array([0, 5.93-2*np.pi, np.pi, 3.734-2*np.pi, 0, 5.408-2*np.pi, np.pi/2])
q_push = np.array([0.0, np.pi/4+0.13, np.pi, 4.4-2*np.pi, 0, 1.2, np.pi/2])
q_wrap = np.pi/180*np.array([55, 125, 114, 244, 217, 45, 8])

# Some useful ball pose definitions
radius = 0.1   # of ball
q_ball_start = np.array([0,0,0,1,0.6,0.0,radius])
q_ball_target = np.array([0,0,0,1,0.6,0.0,radius])
if scenario == "lift":
    q_ball_start[4] = 0.155  # ball starts close to the base
    q_ball_target[6] += 0.2   # goal is to lift it in the air
elif scenario == "forward":
    q_ball_target[4] += 0.2   # goal is to move the ball forward
elif scenario == "side":
    q_ball_target[5] += 0.15  # goal is to move the ball to the side
else:
    raise RuntimeError("Unknown scenario %s"%scenario)

# Initial state
q_start = q_push
if scenario == "lift":
    q_start = q_wrap
x0 = np.hstack([q_start, q_ball_start, np.zeros(13)])

# Target state
x_nom = np.hstack([q_start, q_ball_target, np.zeros(13)])

# Quadratic cost
Qq_robot = 0.0*np.ones(7)
Qv_robot = 0.1*np.ones(7)
Qq_ball = 1*np.array([0,0,0,0,100,100,100])
if scenario == "lift":
    # Don't penalize x and y position for the lifting example
    Qq_ball[4] = 0
    Qq_ball[5] = 0

Qv_ball = 0.1*np.ones(6)
Q_diag = np.hstack([Qq_robot, Qq_ball, Qv_robot, Qv_ball])
Qf_diag = np.hstack([Qq_robot, Qq_ball, Qv_robot, 10*Qv_ball])

Q = np.diag(Q_diag)
R = 0.01*np.eye(7)
Qf = np.diag(Qf_diag)

# Contact model parameters
dissipation = 5.0              # controls "bounciness" of collisions: lower is bouncier
hydroelastic_modulus = 5e6     # controls "squishiness" of collisions: lower is squishier
resolution_hint = 0.05         # smaller means a finer mesh

mu_static = 0.3
mu_dynamic = 0.2

contact_model = ContactModel.kHydroelastic  # Hydroelastic, Point, or HydroelasticWithFallback
mesh_type = HydroelasticContactRepresentation.kTriangle  # Triangle or Polygon

####################################
# Tools for system setup
####################################

def create_system_model(plant, scene_graph):
    # Add the kinova arm model from urdf
    # (rigid hydroelastic contact included)
    urdf = "models/kinova_gen3/urdf/GEN3_URDF_V12.urdf"
    arm = Parser(plant).AddModelFromFile(urdf)
    X_robot = RigidTransform()
    X_robot.set_translation([0,0,0.015])  # base attachment sets the robot up a bit
    plant.WeldFrames(plant.world_frame(),
                     plant.GetFrameByName("base_link", arm),
                     X_robot)
    
    # Add an unactuated gripper from urdf
    # (rigid hydroelastic contact included)
    urdf = "models/2f_85_gripper/urdf/robotiq_2f_85_static.urdf"
    gripper = Parser(plant).AddModelFromFile(urdf)
    X_grip = RigidTransform()
    X_grip.set_rotation(RotationMatrix(RollPitchYaw([0,0,np.pi/2])))
    plant.WeldFrames(plant.GetFrameByName("end_effector_link",arm),
                     plant.GetFrameByName("robotiq_arg2f_base_link", gripper),
                     X_grip)

    # DEBUG: ignore contacts between gripper and robot
    gripper_bodies = [plant.GetBodyByName("robotiq_arg2f_base_link"),
                      plant.GetBodyByName("right_inner_finger"),
                      plant.GetBodyByName("right_outer_finger"),
                      plant.GetBodyByName("right_inner_knuckle"),
                      plant.GetBodyByName("right_outer_knuckle"),
                      plant.GetBodyByName("right_inner_finger_pad"),
                      plant.GetBodyByName("left_inner_finger"),
                      plant.GetBodyByName("left_outer_finger"),
                      plant.GetBodyByName("left_inner_knuckle"),
                      plant.GetBodyByName("left_outer_knuckle"),
                      plant.GetBodyByName("left_inner_finger_pad")]
    arm_bodies = [plant.GetBodyByName("bracelet_with_vision_link"),
                  plant.GetBodyByName("spherical_wrist_2_link"),
                  plant.GetBodyByName("spherical_wrist_1_link"),
                  plant.GetBodyByName("forearm_link"),
                  plant.GetBodyByName("half_arm_1_link"),
                  plant.GetBodyByName("half_arm_2_link"),
                  plant.GetBodyByName("shoulder_link"),
                  plant.GetBodyByName("base_link")]
    gripper_ids = [plant.GetCollisionGeometriesForBody(body)[0] for body in gripper_bodies]
    arm_ids = [plant.GetCollisionGeometriesForBody(body)[0] for body in arm_bodies]

    set_one = GeometrySet(gripper_ids)
    set_two = GeometrySet(arm_ids)

    collision_filter = CollisionFilterDeclaration()
    collision_filter.ExcludeBetween(set_one, set_two)
    scene_graph.collision_filter_manager().Apply(collision_filter)

    # Add a ground with compliant hydroelastic contact
    ground_props = ProximityProperties()
    AddCompliantHydroelasticProperties(resolution_hint, hydroelastic_modulus,ground_props)
    friction = CoulombFriction(0.7*mu_static, 0.7*mu_dynamic)
    AddContactMaterial(dissipation=dissipation, friction=friction, properties=ground_props)
    X_ground = RigidTransform()
    X_ground.set_translation([0,0,-0.5])
    ground_shape = Box(25,25,1)
    plant.RegisterCollisionGeometry(plant.world_body(), X_ground,
            ground_shape, "ground_collision", ground_props)

    # Add a ball with compliant hydroelastic contact
    mass = 0.258
    I = SpatialInertia(mass, np.zeros(3), UnitInertia.HollowSphere(radius))
    ball_instance = plant.AddModelInstance("ball")
    ball = plant.AddRigidBody("ball",ball_instance, I)
    X_ball = RigidTransform()

    ball_props = ProximityProperties()
    AddCompliantHydroelasticProperties(resolution_hint, hydroelastic_modulus,ball_props)
    AddContactMaterial(dissipation=dissipation, friction=friction, properties=ball_props)
    plant.RegisterCollisionGeometry(ball, X_ball, Sphere(radius),
            "ball_collision", ball_props)

    color = np.array([0.8,1.0,0.0,0.5])
    plant.RegisterVisualGeometry(ball, X_ball, Sphere(radius), "ball_visual", color)

    # Add some spots to visualize the ball's roation
    spot_color = np.array([0.0,0.00,0.0,0.5])
    spot_radius = 0.05*radius;
    spot = Sphere(spot_radius);
    spot_offset = radius - 0.45*spot_radius;

    plant.RegisterVisualGeometry(
            ball, RigidTransform(RotationMatrix(),np.array([radius,0,0])),
            spot, "sphere_x+", spot_color)
    plant.RegisterVisualGeometry(
            ball, RigidTransform(RotationMatrix(),np.array([-radius,0,0])),
            spot, "sphere_x-", spot_color)
    plant.RegisterVisualGeometry(
            ball, RigidTransform(RotationMatrix(),np.array([0,radius,0])),
            spot, "sphere_y+", spot_color)
    plant.RegisterVisualGeometry(
            ball, RigidTransform(RotationMatrix(),np.array([0,-radius,0])),
            spot, "sphere_y-", spot_color)
    plant.RegisterVisualGeometry(
            ball, RigidTransform(RotationMatrix(),np.array([0,0,radius])),
            spot, "sphere_z+", spot_color)
    plant.RegisterVisualGeometry(
            ball, RigidTransform(RotationMatrix(),np.array([0,0,-radius])),
            spot, "sphere_z-", spot_color)

    # Choose contact model
    plant.set_contact_surface_representation(mesh_type)
    plant.set_contact_model(contact_model)
    plant.Finalize()

    return plant, scene_graph

####################################
# Create system diagram
####################################
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, dt)
plant, scene_graph = create_system_model(plant, scene_graph)

# Connect to visualizer
if meshcat_visualisation:
    meshcat = StartMeshcat()
    visualizer = MeshcatVisualizer.AddToBuilder( 
        builder, scene_graph, meshcat,
        MeshcatVisualizerParams(role=Role.kPerception, prefix="visual"))
else:
    params = DrakeVisualizerParams(role=Role.kProximity, show_hydroelastic=True)
    DrakeVisualizer(params=params).AddToBuilder(builder, scene_graph)
    ConnectContactResultsToDrakeVisualizer(builder, plant, scene_graph)


# Finailze the diagram
diagram = builder.Build()
diagram_context = diagram.CreateDefaultContext()
plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)

####################################
# Solve Trajectory Optimization
####################################

if optimize:
    # Create a system model (w/o visualizer) to do the optimization over
    builder_ = DiagramBuilder()
    plant_, scene_graph_ = AddMultibodyPlantSceneGraph(builder_, dt)
    plant_, scene_graph_ = create_system_model(plant_, scene_graph_)
    builder_.ExportInput(plant_.get_actuation_input_port(), "control")
    system_ = builder_.Build()

    # Set up the optimizer
    num_steps = int(T/dt)
    # interpolation_method = utils_derivs_interpolation.derivs_interpolation('setInterval', 5, 0, 0, 0)
    # interpolation_method = utils_derivs_interpolation.derivs_interpolation('adaptiveJerk', 5, 40, 1e-3, 0)
    # interpolation_method = utils_derivs_interpolation.derivs_interpolation('setInterval', 1, 0, 0, 0)
    interpolation_method = utils_derivs_interpolation.derivs_interpolation('iterativeError', 5, 0, 0, 10)
    ilqr = IterativeLinearQuadraticRegulator(system_, num_steps, 
            beta=0.5, delta=1e-3, gamma=0, derivs_keypoint_method = interpolation_method)

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
    u_guess = u_gravity_comp
    ilqr.SetInitialGuess(u_guess)

    # Solve the optimization problem
    states, inputs, solve_time, optimal_cost = ilqr.Solve()
    print(f"Solved in {solve_time} seconds using iLQR")
    print(f"Optimal cost: {optimal_cost}")
    timesteps = np.arange(0.0,T,dt)

    # save the solution
    ilqr.SaveSolution(save_file)

#####################################
# Playback
#####################################

if playback:

    if not optimize:
        # load previously computed solution from file
        data = np.load(save_file)
        timesteps = data["t"]
        states = data["x_bar"]

    while True:
        plant.get_actuation_input_port().FixValue(plant_context, 
                np.zeros(plant.num_actuators()))
        # Just keep playing back the trajectory
        for i in range(len(timesteps)):
            t = timesteps[i]
            x = states[:,i]

            diagram_context.SetTime(t)
            plant.SetPositionsAndVelocities(plant_context, x)
            diagram.ForcedPublish(diagram_context)

            time.sleep(1/playback_rate*dt-4e-4)
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
    simulator.set_publish_every_time_step(True)

    simulator.AdvanceTo(T)
