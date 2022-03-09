#include <iostream>

#include <drake/geometry/scene_graph.h>
#include <drake/multibody/parsing/parser.h>
#include <drake/multibody/plant/multibody_plant.h>
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include <drake/systems/framework/context.h>
#include <drake/systems/framework/diagram_builder.h>

/**
 *
 * Run some tests for a simple ball
 * with hydroelastic contact.
 *
 */

using namespace drake;

using systems::Context;
using systems::Diagram;
using systems::DiagramBuilder;
using multibody::MultibodyPlant;
using multibody::MultibodyPlantConfig;
using geometry::SceneGraph;

int main() {

  // Plant parameters
  MultibodyPlantConfig config;
  config.time_step = 1e-2;
  config.contact_model = "hydroelastic";
  config.contact_surface_representation = "polygon";

  // Ball parameters
  double radius = 0.05;
  double mass = 0.1;

  // Contact parameters
  double dissipation = 0.0;
  double hydroelastic_modulus = 5e4;
  double resolution_hint = 0.002;
  double mu = 0.7;

  // Set up the system
  DiagramBuilder<double> builder;
  auto [plant, scene_graph] = AddMultibodyPlant(config, &builder);

  std::cout << "Hello World!" << std::endl;
  return 0;
}
