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
  const double resolution_hint = 0.003;
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
  q0 << 0, 0, 0, -1, -0.5, 0, 0.047;
  v0 << 0, 0, 0, 0.5, 0, 0;
  VectorXd x0(13);
  x0 << q0, v0;
  diagram_context->SetDiscreteState(x0);

  // Timer variables
  auto st = std::chrono::high_resolution_clock::now();
  auto et = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> elapsed;

  // Simulate forward one step
  //std::unique_ptr<DiscreteValues<double>> update = plant.AllocateDiscreteVariables();
  //st = std::chrono::high_resolution_clock::now();
  //plant.CalcDiscreteVariableUpdates(plant_context, update.get());
  //et = std::chrono::high_resolution_clock::now();
  //elapsed = et - st;
  //std::cout << "Computed forward dynamics in " << elapsed.count() << "s" << std::endl;

  //VectorXd x_next = update->get_mutable_value();
  //std::cout << x_next << std::endl;

  // Compute dynamics gradients with autodiff
  auto system_ad = diagram->ToAutoDiffXd();
  auto context_ad = system_ad->CreateDefaultContext();
  auto x_ad = InitializeAutoDiff(x0);
  context_ad->SetDiscreteState(x_ad);
  std::unique_ptr<DiscreteValues<AutoDiffXd>> update_ad = system_ad->AllocateDiscreteVariables();

  st = std::chrono::high_resolution_clock::now();
  system_ad->CalcDiscreteVariableUpdates(*context_ad, update_ad.get());
  et = std::chrono::high_resolution_clock::now();
  elapsed = et - st;
  std::cout << "Computed autodiff gradients in " << elapsed.count() << "s" << std::endl;

  auto x_next_ad = update_ad->get_mutable_value();
  auto fx_ad = ExtractGradient(x_next_ad);

  // Compute approximate dynamics gradients with our method
  Eigen::VectorXd x_next(13);
  Eigen::MatrixXd fx(13,13);
  Eigen::MatrixXd fu(13,0);

  st = std::chrono::high_resolution_clock::now();
  plant.DiscreteDynamicsWithApproximateGradients(plant_context, &x_next, &fx, &fu);
  et = std::chrono::high_resolution_clock::now();
  elapsed = et - st;
  std::cout << "Computed forward dynamics with approx gradient in " << elapsed.count() << "s" << std::endl;
 
  // Compare autodiff and approximate gradients
  auto dq_dq_ad = fx_ad.topLeftCorner(7,7);
  auto dq_dv_ad = fx_ad.topRightCorner(7,6);
  auto dv_dq_ad = fx_ad.bottomLeftCorner(6,7);
  auto dv_dv_ad = fx_ad.bottomRightCorner(6,6);
  
  auto dq_dq = fx.topLeftCorner(7,7);
  auto dq_dv = fx.topRightCorner(7,6);
  auto dv_dq = fx.bottomLeftCorner(6,7);
  auto dv_dv = fx.bottomRightCorner(6,6);

  //std::cout << "dq_dq" << std::endl;
  //std::cout << dq_dq_ad << std::endl;
  //std::cout << std::endl;
  //std::cout << dq_dq << std::endl;
  //std::cout << std::endl;
  //std::cout << std::endl;
  
  //std::cout << "dq_dv" << std::endl;
  //std::cout << dq_dv_ad << std::endl;
  //std::cout << std::endl;
  //std::cout << dq_dv << std::endl;
  //std::cout << std::endl;
  //std::cout << std::endl;
  
  std::cout << "dv_dq" << std::endl;
  std::cout << dv_dq_ad << std::endl;
  std::cout << std::endl;
  std::cout << dv_dq << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
  
  //std::cout << "dv_dv" << std::endl;
  //std::cout << dv_dv_ad << std::endl;
  //std::cout << std::endl;
  //std::cout << dv_dv << std::endl;
  //std::cout << std::endl;
  //std::cout << std::endl;
  
  return 0;
}
