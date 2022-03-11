#include <iostream>
#include <chrono>

#include <drake/geometry/proximity_properties.h>
#include <drake/geometry/scene_graph.h>
#include <drake/multibody/parsing/parser.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/multibody/plant/multibody_plant_config_functions.h>
#include <drake/multibody/plant/tamsi_solver.h>
#include <drake/systems/framework/context.h>
#include <drake/systems/framework/diagram_builder.h>

/**
 *
 * Run some tests for a simple ball
 * with hydroelastic contact.
 *
 */

using namespace drake;

using Eigen::VectorXd;
using geometry::AddCompliantHydroelasticProperties;
using geometry::AddContactMaterial;
using geometry::AddRigidHydroelasticProperties;
using geometry::HalfSpace;
using geometry::ProximityProperties;
using geometry::SceneGraph;
using geometry::Sphere;
using math::RigidTransformd;
using math::ExtractGradient;
using math::InitializeAutoDiff;
using multibody::CoulombFriction;
using multibody::MultibodyPlant;
using multibody::MultibodyPlantConfig;
using multibody::RigidBody;
using multibody::SpatialInertia;
using multibody::TamsiSolver;
using multibody::TamsiSolverResult;
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
  const double radius = 0.05;
  const double mass = 0.1;

  // Contact parameters
  const double dissipation = 0.0;
  const double hydroelastic_modulus = 5e4;
  const double resolution_hint = 0.002;
  const double mu = 0.7;
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
  AddCompliantHydroelasticProperties(resolution_hint, hydroelastic_modulus, &ball_props);
  AddContactMaterial(dissipation, {}, surface_friction, &ball_props);
  plant.RegisterCollisionGeometry(ball, X_BS, Sphere(radius), "ball_collision",
                                     std::move(ball_props));

  const Vector4<double> orange(1.0, 0.55, 0.0, 1.0);
  plant.RegisterVisualGeometry(ball, X_BS, Sphere(radius), "visual", orange);

  // Add a ground with rigid hydroelastic contact
  const RigidTransformd X_WS;
  ProximityProperties ground_props;
  AddRigidHydroelasticProperties(&ground_props);
  AddContactMaterial(dissipation, {}, surface_friction, &ground_props);
  plant.RegisterCollisionGeometry(plant.world_body(), X_WS, HalfSpace{}, "collision",
      std::move(ground_props));

  // Compile the system diagram
  plant.Finalize();
  auto diagram = builder.Build();
  std::unique_ptr<Context<double>> diagram_context = diagram->CreateDefaultContext();
  Context<double>& plant_context = 
    diagram->GetMutableSubsystemContext(plant, diagram_context.get());

  // Set initial conditions
  VectorXd q0(7);
  VectorXd v0(6);
  q0 << 0, 0, 0, 1, -0.5, 0, 0.047;
  v0 << 0, 0, 0, 0.5, 0, 0;
  VectorXd x0(13);
  x0 << q0, v0;
  diagram_context->SetDiscreteState(x0);

  // Simulate forward one step
  std::unique_ptr<DiscreteValues<double>> update = plant.AllocateDiscreteVariables();
  
  auto st = std::chrono::high_resolution_clock::now();
  plant.CalcDiscreteVariableUpdates(plant_context, update.get());
  auto et = std::chrono::high_resolution_clock::now();

  VectorXd x_next = update->get_mutable_value();
  std::chrono::duration<float> elapsed = et - st;

  std::cout << "Computed forward dynamics in " << elapsed.count() << "s" << std::endl;
  std::cout << x_next << std::endl;

  // Make autodiff plant and compute dynamics gradients
  auto system_ad = diagram->ToAutoDiffXd();
  auto context_ad = system_ad->CreateDefaultContext();
  auto x_ad = InitializeAutoDiff(x0);
  context_ad->SetDiscreteState(x_ad);
  std::unique_ptr<DiscreteValues<AutoDiffXd>> update_ad = system_ad->AllocateDiscreteVariables();

  auto st_ad = std::chrono::high_resolution_clock::now();
  system_ad->CalcDiscreteVariableUpdates(*context_ad, update_ad.get());
  auto et_ad = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> elapsed_ad = et_ad - st_ad;
  std::cout << "Computed autodiff gradients in " << elapsed_ad.count() << "s" << std::endl;

  auto x_next_ad = update_ad->get_mutable_value();
  auto fx = ExtractGradient(x_next_ad);
  std::cout << fx << std::endl;

  // DEBUG
  //TamsiSolver<double> tamsi_solver(plant.num_velocities());
  //const VectorXd v_guess(6);
  //TamsiSolverResult result = tamsi_solver.SolveWithGuess(1e-2, v_guess);
  //std::cout << tamsi_solver.get_generalized_friction_forces() << std::endl;

  return 0;
}
