#!/usr/bin/env python

##
#
# Testing uncertainty propagation using a simple pendulum model. 
#
##

from pydrake.all import *
import time
import matplotlib.pyplot as plt

def create_system_model(builder):
    """
    Create and return a plant and scene graph for the simple pendulum system.
    """
    # System parameters
    dt = 1e-3
    mass = 1.0
    radius = 0.01
    length = 0.5
    damping = 0.1

    # Create the plant and scene graph
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, dt)

    # Create the pendulum object
    pendulum = plant.AddModelInstance("pendulum")
    rod = plant.AddRigidBody("rod", pendulum, SpatialInertia(mass, [0,0,0],
                UnitInertia.SolidCylinder(radius, length)))
    rod_com_frame = plant.GetFrameByName("rod")
    X_base_com = RigidTransform(RotationMatrix(),[0,0,length/2])
    rod_base_frame = plant.AddFrame(FixedOffsetFrame("rod_base",
        rod_com_frame, X_base_com))
    plant.AddJoint(RevoluteJoint("base_joint", plant.world_frame(),
        rod_base_frame, [1,0,0], damping))
    
    rod_shape = Cylinder(radius, length)

    rod_props = ProximityProperties()
    AddCompliantHydroelasticProperties(radius, 5e6, rod_props)
    AddContactMaterial(friction=CoulombFriction(0.5,0.5), properties=rod_props)
    plant.RegisterCollisionGeometry(rod, RigidTransform(), rod_shape,
            "rod_collision", rod_props)

    plant.RegisterVisualGeometry(rod, RigidTransform(), rod_shape, "rod_visual", 
            [0.5,0.5,0.9,1.0])

    ## Add a box to collide with
    #box = plant.AddModelInstance("box")
    #box_body = plant.AddRigidBody("box", box, 
    #        SpatialInertia(1.0, [0,0,0], UnitInertia.SolidBox(1,1,1)))
    #X_box = RigidTransform(RotationMatrix(),[0,0.5,0])
    #box_frame = plant.GetFrameByName("box")
    #plant.WeldFrames(plant.world_frame(), box_frame, X_box)

    #box_shape = Box(0.2,0.2,0.2)

    #box_props = ProximityProperties()
    #AddCompliantHydroelasticProperties(1.0, 5e5, box_props)
    #AddContactMaterial(dissipation=1, friction=CoulombFriction(0.5,0.5), properties=box_props)
    #plant.RegisterCollisionGeometry(box_body, RigidTransform(), box_shape,
    #        "box_collision", box_props)

    #plant.RegisterVisualGeometry(box_body, RigidTransform(), box_shape, 
    #        "box_visual", [0.5,0.9,0.5,1.0])

    plant.Finalize()

    return plant, scene_graph

def plot_mean_and_covariance(t, mu, Sigma):
    """
    Make a plot of the mean state (theta, theta_dot) together
    with its covariance estimate as a green interval. 
    """
    # theta
    plt.subplot(2,1,1)
    plt.plot(t,mu[0,:])
    covariance_lb = mu[0,:] - Sigma[0,0,:]
    covariance_ub = mu[0,:] + Sigma[0,0,:]
    plt.fill_between(t, covariance_lb, covariance_ub, color='green', alpha=0.5)
    plt.ylabel("theta")

    # theta dot
    plt.subplot(2,1,2)
    plt.plot(t,mu[1,:])
    covariance_lb = mu[1,:] - Sigma[1,1,:]
    covariance_ub = mu[1,:] + Sigma[1,1,:]
    plt.fill_between(t, covariance_lb, covariance_ub, color='green', alpha=0.5)
    plt.ylabel("theta dot")
    plt.xlabel("time (s)")

if __name__=="__main__":
    # Create the system model
    builder = DiagramBuilder()
    plant, scene_graph = create_system_model(builder)

    DrakeVisualizer().AddToBuilder(builder, scene_graph)
    #ConnectContactResultsToDrakeVisualizer(builder, plant, scene_graph)

    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)

    # Create an autodiff copy of the plant
    diagram_ad = diagram.ToAutoDiffXd()
    diagram_context_ad = diagram_ad.CreateDefaultContext()

    # Set initial conditions
    x0 = np.array([0.8*np.pi,0])
    plant.SetPositionsAndVelocities(plant_context, x0)

    # Set up recording of state trajectory
    T = 2.0
    N = int(T/plant.time_step())
    n = plant.num_multibody_states()
    x = np.zeros((n, N))
    
    Sigma = np.zeros((n,n,N+1))
    Sigma0 = np.diag([0.1,0.1])
    Sigma[:,:,0] = Sigma0

    # Step through a simulation
    t = 0.0
    for i in range(N):
        st = time.time()

        # Record the current state x_k
        x[:,i] = plant.GetPositionsAndVelocities(plant_context)

        # Compute the discrete state update x_{k+1} = f(x_k)
        state = diagram_context.get_mutable_discrete_state()
        diagram.CalcDiscreteVariableUpdates(diagram_context, state)

        # Compute the derivatives of the discrete state update f_x
        x_ad = InitializeAutoDiff(x[:,i])
        diagram_context_ad.SetDiscreteState(x_ad)
        state_ad = diagram_context_ad.get_discrete_state()
        diagram_ad.CalcDiscreteVariableUpdates(diagram_context_ad, state_ad)
        x_next_ad = state_ad.get_vector().CopyToVector()
        fx = ExtractGradient(x_next_ad)

        # Use these derivatives to update the state covariance estimate
        Sigma[:,:,i+1] = fx@Sigma[:,:,i]@fx.T

        # Set the time and publish (so the visualizer gets the new state)
        diagram_context.SetTime(t)
        diagram.Publish(diagram_context)
      
        # Update timestep and try to match real-time rate
        t += plant.time_step()
        sleep_time = st-time.time()+plant.time_step()-2e-4
        #time.sleep(max(0,sleep_time))

    # Sample from initial distribution and simulate
    ns = 10
    xs = np.zeros((n,N,ns))
    for j in range(ns):
        x0_j = np.random.multivariate_normal(x0, Sigma0)
        plant.SetPositionsAndVelocities(plant_context, x0_j)
        for i in range(N):
                xs[:,i,j] = plant.GetPositionsAndVelocities(plant_context)
                state = diagram_context.get_mutable_discrete_state()
                diagram.CalcDiscreteVariableUpdates(diagram_context, state)

    # Compute sample mean and covariance
    mu_sampled = np.mean(xs, axis=2)
    Sigma_sampled = np.zeros((n,n,N))
    for i in range(N):
        summ = 0
        for j in range(ns):
            diff = (xs[:,i,j] - mu_sampled[:,i])[np.newaxis].T
            Sigma_sampled[:,:,i] += diff@diff.T
        Sigma_sampled[:,:,i] *= 1/(ns-1)
   
    # Make plots
    t = np.arange(0,T,plant.time_step())

    plt.figure("Local Estimate")
    plot_mean_and_covariance(t, x, Sigma[:,:,:-1])

    plt.figure("Samples")
    plt.subplot(2,1,1)
    plt.plot(t,xs[0,:,:])
    plt.subplot(2,1,2)
    plt.plot(t,xs[1,:,:])

    plt.figure("MC Estimate")
    plot_mean_and_covariance(t, mu_sampled, Sigma_sampled)
    
    plt.show()

