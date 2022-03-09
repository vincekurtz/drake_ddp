#include <iostream>

#include <drake/geometry/proximity_properties.h>
#include <drake/geometry/scene_graph.h>
#include <drake/multibody/parsing/parser.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/multibody/plant/multibody_plant_config_functions.h>
#include <drake/systems/framework/context.h>
#include <drake/systems/framework/diagram_builder.h>

/**
 *
 * Run some tests for a simple ball
 * with hydroelastic contact.
 *
 */

using namespace drake;

using geometry::AddCompliantHydroelasticProperties;
using geometry::AddContactMaterial;
using geometry::AddRigidHydroelasticProperties;
using geometry::ProximityProperties;
using geometry::SceneGraph;
using geometry::Sphere;
using math::RigidTransformd;
using multibody::CoulombFriction;
using multibody::MultibodyPlant;
using multibody::MultibodyPlantConfig;
using multibody::RigidBody;
using multibody::SpatialInertia;
using multibody::UnitInertia;
using systems::Context;
using systems::Diagram;
using systems::DiagramBuilder;
using systems::DiscreteValues;

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
  const CoulombFriction<double> surface_friction(mu, mu);

  // Set up the system
  DiagramBuilder<double> builder;
  auto [plant, scene_graph] = AddMultibodyPlant(config, &builder);

  // Add a ball with compliant hydroelastic contact
  UnitInertia<double> G_Bcm = UnitInertia<double>::SolidSphere(radius);
  SpatialInertia<double> M_Bcm(mass, Vector3<double>::Zero(), G_Bcm);
  const RigidBody<double>& ball = plant.AddRigidBody("Ball", M_Bcm);

  const RigidTransformd X_BS;  // identity.
  ProximityProperties ball_props;
  AddCompliantHydroelasticProperties(radius, hydroelastic_modulus, &ball_props);
  AddContactMaterial(dissipation, {}, surface_friction, &ball_props);
  plant.RegisterCollisionGeometry(ball, X_BS, Sphere(radius), "collision",
                                     std::move(ball_props));

  const Vector4<double> orange(1.0, 0.55, 0.0, 1.0);
  plant.RegisterVisualGeometry(ball, X_BS, Sphere(radius), "visual", orange);

  // Add a ground with rigid hydroelastic contact

  // Compile the system diagram
  plant.Finalize();
  auto diagram = builder.Build();
  std::unique_ptr<Context<double>> diagram_context = diagram->CreateDefaultContext();

  // Set initial conditions


  // Simulate forward one step
  std::unique_ptr<DiscreteValues<double>> update = diagram->AllocateDiscreteVariables();
  //update->SetFrom(diagram_context->get_mutable_discrete_state());
  diagram->CalcDiscreteVariableUpdates(*diagram_context, update.get());
  auto x_next = update->get_mutable_value();

  std::cout << x_next << std::endl;
  return 0;
}
