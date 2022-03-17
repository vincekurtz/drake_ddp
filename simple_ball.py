#!/usr/bin/env python

##
#
# Simple example of an unactuated ball for testing
# hydroelastic contact ideas
#
##

from pydrake.all import *
import time

# Some simulation parameters 
T = 5.0
dt = 0
realtime_rate = 0.0

radius = 0.05
mass = 0.1

dissipation = 0.0              # controls "bounciness" of collisions: lower is bouncier
hydroelastic_modulus = 5e4     # controls "squishiness" of collisions: lower is squishier
resolution_hint = 0.003         # smaller means a finer mesh
mu = 0.7                       # friction coefficient, same for static and dynamic
surface_type = HydroelasticContactRepresentation.kPolygon  # Polygon or Triangle

# Set up system diagram
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, dt)

def create_system_model(plant, scene_graph):
    # Add a ball with compliant hydroelastic contact
    G_Bcm = UnitInertia.SolidSphere(radius)
    M_Bcm = SpatialInertia(mass, np.zeros(3), G_Bcm)
    ball = plant.AddRigidBody("ball", M_Bcm)

    X_BS = RigidTransform()
    ball_props = ProximityProperties()
    AddCompliantHydroelasticProperties(resolution_hint, hydroelastic_modulus, ball_props)
    AddContactMaterial(dissipation=dissipation, friction=CoulombFriction(mu,mu), properties=ball_props)
    plant.RegisterCollisionGeometry(ball, X_BS, Sphere(radius),
            "ball_collision", ball_props)

    orange = np.array([1.0, 0.55, 0.0, 0.5])
    plant.RegisterVisualGeometry(ball, X_BS, Sphere(radius), "visual", orange)
        
    # Add some spots to visualize the ball's roation
    spot_color = np.array([0.0,0.00,0.0,0.5])
    spot_radius = 0.1*radius
    spot = Sphere(spot_radius)
    spot_offset = radius - 0.9*spot_radius;

    plant.RegisterVisualGeometry(
            ball, RigidTransform(RotationMatrix(),np.array([spot_offset,0,0])),
            spot, "sphere_x+", spot_color)
    plant.RegisterVisualGeometry(
            ball, RigidTransform(RotationMatrix(),np.array([-spot_offset,0,0])),
            spot, "sphere_x-", spot_color)
    plant.RegisterVisualGeometry(
            ball, RigidTransform(RotationMatrix(),np.array([0,spot_offset,0])),
            spot, "sphere_y+", spot_color)
    plant.RegisterVisualGeometry(
            ball, RigidTransform(RotationMatrix(),np.array([0,-spot_offset,0])),
            spot, "sphere_y-", spot_color)
    plant.RegisterVisualGeometry(
            ball, RigidTransform(RotationMatrix(),np.array([0,0,spot_offset])),
            spot, "sphere_z+", spot_color)
    plant.RegisterVisualGeometry(
            ball, RigidTransform(RotationMatrix(),np.array([0,0,-spot_offset])),
            spot, "sphere_z-", spot_color)

    # Add a ground with rigid hydroelastic contact
    ground_props = ProximityProperties()
    AddRigidHydroelasticProperties(ground_props)
    AddContactMaterial(dissipation=dissipation, friction=CoulombFriction(mu, mu), properties=ground_props)
    plant.RegisterCollisionGeometry(
            plant.world_body(), RigidTransform(), HalfSpace(), 
            "ground_collision", ground_props)

    # Turn off gravity
    #plant.mutable_gravity_field().set_gravity_vector([0,0,0])

    # Choose contact model
    plant.set_contact_surface_representation(surface_type)
    plant.set_contact_model(ContactModel.kHydroelastic)
    plant.Finalize()

# Add the system model
create_system_model(plant, scene_graph)

# Connect to visualizer
DrakeVisualizer().AddToBuilder(builder, scene_graph)
ConnectContactResultsToDrakeVisualizer(builder, plant, scene_graph)

# Finalize the diagram
diagram = builder.Build()
diagram_context = diagram.CreateDefaultContext()
plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)

# Set initial conditions
q0 = np.array([0,0,0,1,-0.5,0,0.047])
v0 = np.array([0,0,0,0.05,0,0])
plant.SetPositions(plant_context, q0)
plant.SetVelocities(plant_context, v0)

# Simulate the sytem
simulator = Simulator(diagram, diagram_context)

print(GetIntegrationSchemes())
ResetIntegratorFromFlags(simulator, "implicit_euler", 1e-2)

simulator.Initialize()
integrator = simulator.get_integrator()
print(integrator)
integrator.set_fixed_step_mode(True)
print(integrator.get_fixed_step_mode())
simulator.set_target_realtime_rate(realtime_rate)
simulator.AdvanceTo(T)
