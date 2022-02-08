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

# Contact model parameters
dissipation = 5.0              # controls "bounciness" of collisions: lower is bouncier
hydroelastic_modulus = 5e4     # controls "squishiness" of collisions: lower is squishier
resolution_hint = 0.05         # smaller means a finer mesh

# Set up system diagram
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, dt)

# Add the cart-pole system
sdf = FindResourceOrThrow("drake/examples/multibody/cart_pole/cart_pole.sdf")
robot = Parser(plant=plant).AddModelFromFile(sdf)

# Add a ball with compliant hydroelastic contact to the end of the cart-pole system
pole = plant.GetBodyByName("Pole")
X_BP = RigidTransform()
ball_props = ProximityProperties()
AddCompliantHydroelasticProperties(resolution_hint, hydroelastic_modulus, ball_props)
AddContactMaterial(dissipation=dissipation, friction=CoulombFriction(), properties=ball_props)
plant.RegisterCollisionGeometry(pole, X_BP, Sphere(radius), "collision", ball_props)

orange = np.array([1.0, 0.55, 0.0, 0.5])
plant.RegisterVisualGeometry(pole, X_BP, Sphere(radius), "visual", orange)

# Add a wall with rigid hydroelastic contact
l,w,h = (0.05,1,2)   
I_W = SpatialInertia(1, np.zeros(3), UnitInertia.SolidBox(l,w,h))
wall_instance = plant.AddModelInstance("wall")
wall = plant.AddRigidBody("wall", wall_instance, I_W)
wall_frame = plant.GetFrameByName("wall", wall_instance)
X_W = RigidTransform()
X_W.set_translation([-0.3,0,0])
plant.WeldFrames(plant.world_frame(), wall_frame, X_W)

plant.RegisterVisualGeometry(wall, RigidTransform(), Box(l,w,h), "wall_visual", orange)

wall_props = ProximityProperties()
AddRigidHydroelasticProperties(wall_props)
AddContactMaterial(dissipation=dissipation, friction=CoulombFriction(), properties=wall_props)
plant.RegisterCollisionGeometry(wall, RigidTransform(), 
        Box(l,w,h), "wall_collision", wall_props)

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

# Fix zero input for now
plant.get_actuation_input_port().FixValue(plant_context, 0)

# Set initial state
q0 = np.array([0,np.pi-0.2])
v0 = np.array([0,0])
plant.SetPositions(plant_context, q0)
plant.SetVelocities(plant_context, v0)

# Simulate the system
simulator = Simulator(diagram, diagram_context)
simulator.set_target_realtime_rate(1.0)

simulator.AdvanceTo(T)
