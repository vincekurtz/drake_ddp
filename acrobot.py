#!/usr/bin/env python

##
#
# Do swing-up control of an acrobot
#
##

import numpy as np
from pydrake.all import *
import time

####################################
# Parameters
####################################

T = 10         # total simulation time (S)
dt = 1e-3      # simulation timestep

# CONST = constant (zero) torque
# DT = direct transcription
# DDP = differential dynamic programming
control_method = "CONST" 

# Initial state
x0 = np.array([0,np.pi/2,0,0])

####################################
# Create system diagram
####################################
builder = DiagramBuilder()

plant, scene_graph = AddMultibodyPlantSceneGraph(builder, dt)
urdf = FindResourceOrThrow("drake/examples/acrobot/Acrobot.urdf")
robot = Parser(plant=plant).AddModelFromFile(urdf)

plant.Finalize()
assert plant.geometry_source_is_registered()

if control_method == "CONST":
    controller = builder.AddSystem(ConstantVectorSource(np.zeros(1)))
elif control_method == "DT":
    pass
elif control_method == "DDP":
    pass
else:
    raise ValueError("Unrecognized control method %s"%control_method)

builder.Connect(
        controller.get_output_port(),
        plant.get_actuation_input_port())

logger = LogVectorOutput(plant.get_state_output_port(), builder)

DrakeVisualizer().AddToBuilder(builder, scene_graph)
ConnectContactResultsToDrakeVisualizer(builder, plant, scene_graph)

diagram = builder.Build()
diagram_context = diagram.CreateDefaultContext()

####################################
# Run simulation
####################################
simulator = Simulator(diagram, diagram_context)
simulator.set_target_realtime_rate(-1.0)

plant_context = diagram.GetMutableSubsystemContext(
        plant, simulator.get_mutable_context())
plant.SetPositionsAndVelocities(plant_context, x0)

simulator.AdvanceTo(T)

#####################################
# Playback on visualizer
#####################################
log = logger.FindLog(diagram_context)
states = log.data()
timesteps = log.sample_times()

for i in range(len(timesteps)):
    t = timesteps[i]
    x = states[:,i]

    diagram_context.SetTime(t)
    plant.SetPositionsAndVelocities(plant_context, x)
    diagram.Publish(diagram_context)

    time.sleep(dt-3e-4)
