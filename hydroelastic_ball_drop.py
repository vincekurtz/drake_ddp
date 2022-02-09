#!/usr/bin/env python

##
#
# Simple simulation to test Drake's new hydroelastic 
# contact model.
#
##

from pydrake.all import *

# Some simulation parameters 
T = 2.0
dt = 1e-3

radius = 0.05
mass = 0.1

dissipation = 0.0              # controls "bounciness" of collisions: lower is bouncier
hydroelastic_modulus = 5e4     # controls "squishiness" of collisions: lower is squishier
resolution_hint = 0.05         # smaller means a finer mesh

# Set up system diagram
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, dt)

# Add a ball with compliant hydroelastic contact
#G_Bcm = UnitInertia.SolidSphere(radius)
#M_Bcm = SpatialInertia(mass, np.zeros(3), G_Bcm)
#ball = plant.AddRigidBody("ball", M_Bcm)
#
#X_BS = RigidTransform()
#ball_props = ProximityProperties()
#AddCompliantHydroelasticProperties(resolution_hint, hydroelastic_modulus, ball_props)
#AddContactMaterial(dissipation=dissipation, friction=CoulombFriction(), properties=ball_props)
#plant.RegisterCollisionGeometry(ball, X_BS, Sphere(radius),
#        "ball_collision", ball_props)
#
#orange = np.array([1.0, 0.55, 0.0, 0.5])
#plant.RegisterVisualGeometry(ball, X_BS, Sphere(radius), "visual", orange)

# Add a ball with compliant hydroelastic contact from file
ball = Parser(plant).AddModelFromFile("models/sphere/sphere.urdf")

# Add a ground with rigid hydroelastic contact
ground_props = ProximityProperties()
AddRigidHydroelasticProperties(ground_props)
AddContactMaterial(dissipation=dissipation, friction=CoulombFriction(), properties=ground_props)
plant.RegisterCollisionGeometry(
        plant.world_body(), RigidTransform(), HalfSpace(), 
        "ground_collision", ground_props)

# Choose contact model
plant.set_contact_surface_representation(
        HydroelasticContactRepresentation.kPolygon)
plant.set_contact_model(ContactModel.kHydroelastic)
plant.Finalize()

# Connect to visualizer
DrakeVisualizer().AddToBuilder(builder, scene_graph)
ConnectContactResultsToDrakeVisualizer(builder, plant, scene_graph)

# Finailze the diagram
diagram = builder.Build()
diagram_context = diagram.CreateDefaultContext()
plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)

# Simulate the system
q0 = np.array([0,0,0,1,0,0,0.5])
v0 = np.array([0,0,0,-0.1,0,0.1])
plant.SetPositions(plant_context, q0)
plant.SetVelocities(plant_context, v0)

simulator = Simulator(diagram, diagram_context)
simulator.set_target_realtime_rate(1.0)

simulator.AdvanceTo(T)
