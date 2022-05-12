##
#
# An experimental solver based on Pontryagin's minimum principle
#
##

from pydrake.all import *
import time
import numpy as np

class PontryaginOptimizer():
    """
    Set up and solve a trajectory optimization problem of the form

        min_{u} sum{ (x-x_nom)'Q(x-x_nom) + u'Ru } + (x-x_nom)'Qf(x-x_nom)
        s.t.    x_{t+1} = f(x_t, u_t)

    using an experimental Pontryagin-based optimization method.
    """
    def __init__(self, system, num_timesteps, input_port_index=0):
        """
        Args:
            system:             Drake System describing the discrete-time dynamics
                                 x_{t+1} = f(x_t,u_t). Must be discrete-time.
            num_timesteps:      Number of timesteps to consider in the optimization.
            input_port_index:   InputPortIndex for the control input u_t. Default is to
                                 use the first port. 
        """
        assert system.IsDifferenceEquationSystem()[0],  "must be a discrete-time system"

        # float-type copy of the system and context for linesearch.
        # Computations using this system are fast but don't support gradients
        self.system = system
        self.context = self.system.CreateDefaultContext()
        self.input_port = self.system.get_input_port(input_port_index)

        # Make an autodiff copy of the system for computing dynamics gradients
        self.system_ad = system.ToAutoDiffXd()
        self.context_ad = self.system_ad.CreateDefaultContext()
        self.input_port_ad = self.system_ad.get_input_port(input_port_index)
        
        # Extract the MultibodyPlant model from the given system. 
        # The system in this case must include a MultibodyPlant called "plant"
        # which is attached to a corresponding scene graph (for geometry computations)
        self.plant = self.system.GetSubsystemByName("plant")
        self.plant_context = self.system.GetMutableSubsystemContext(self.plant, self.context)
           
        # Set some parameters
        self.N = num_timesteps

        # Define state and input sizes
        self.n = self.context.get_discrete_state_vector().size()
        self.m = self.input_port.size()

        # Initial and target states
        self.x0 = np.zeros(self.n)
        self.x_xom = np.zeros(self.n)

        # Quadratic cost terms
        self.Q = np.eye(self.n)
        self.R = np.eye(self.m)
        self.Qf = np.eye(self.n)

        # Arrays to store best guess of control and state trajectory
        self.x_bar = np.zeros((self.n,self.N))
        self.u_bar = np.zeros((self.m,self.N-1))

        # Arrays to store dynamics gradients
        self.fx = np.zeros((self.n,self.n,self.N-1))
        self.fu = np.zeros((self.n,self.m,self.N-1))

        # Array to store (best buess of) costate trajectory
        self.costate = np.zeros((self.n, self.N))

    def SetInitialState(self, x0):
        """
        Fix the initial condition for the optimization.

        Args:
            x0: Vector containing the initial state of the system
        """
        self.x0 = x0

    def SetTargetState(self, x_nom):
        """
        Fix the target state that we're trying to drive the system to.

        Args:
            x_nom:  Vector containing the target system state
        """
        self.x_nom = np.asarray(x_nom).reshape((self.n,))

    def SetRunningCost(self, Q, R):
        """
        Set the quadratic running cost

            (x-x_nom)'Q(x-x_nom) + u'Ru

        Args:
            Q:  The (n,n) state penalty matrix
            R:  The (m,m) control penalty matrix
        """
        assert Q.shape == (self.n,self.n)
        assert R.shape == (self.m,self.m)

        self.Q = Q
        self.R = R

    def SetTerminalCost(self, Qf):
        """
        Set the terminal cost

            (x-x_nom)'Qf(x-x_nom)

        Args:
            Qf: The (n,n) final state penalty matrix
        """
        assert Qf.shape == (self.n, self.n)
        self.Qf = Qf
    
    def SetInitialGuess(self, u_guess):
        """
        Set the initial guess of control tape.

        Args:
            u_guess:    (m,N-1) numpy array containing u at each timestep
        """
        assert u_guess.shape == (self.m, self.N-1)
        self.u_bar = u_guess

    def _running_cost_partials(self, x, u):
        """
        Return the partial derivatives of the (quadratic) running cost

            l = x'Qx + u'Ru

        for the given state and input values.

        Args:
            x:  numpy array representing state
            u:  numpy array representing control

        Returns:
            lx:     1st order partial w.r.t. x
            lu:     1st order partial w.r.t. u
            lxx:    2nd order partial w.r.t. x
            luu:    2nd order partial w.r.t. u
            lux:    2nd order partial w.r.t. u and x
        """
        lx = 2*self.Q@x - 2*self.x_nom.T@self.Q
        lu = 2*self.R@u
        lxx = 2*self.Q
        luu = 2*self.R
        lux = np.zeros((self.m,self.n))

        return (lx, lu, lxx, luu, lux)

    def _terminal_cost_partials(self, x):
        """
        Return the partial derivatives of the (quadratic) terminal cost

            lf = x'Qfx

        for the given state values. 

        Args:
            x: numpy array representing state

        Returns:
            lf_x:   gradient of terminal cost
            lf_xx:  hessian of terminal cost
        """
        lf_x = 2*self.Qf@x - 2*self.x_nom.T@self.Qf
        lf_xx = 2*self.Qf

        return (lf_x, lf_xx)
    
    def _calc_dynamics(self, x, u):
        """
        Given a system state (x) and a control input (u),
        compute the next state 

            x_next = f(x,u)

        Args:   
            x:  An (n,) numpy array representing the state
            u:  An (m,) numpy array representing the control input

        Returns:
            x_next: An (n,) numpy array representing the next state
        """
        # Set input and state variables in our stored model accordingly
        self.context.SetDiscreteState(x)
        self.input_port.FixValue(self.context, u)

        # Compute the forward dynamics x_next = f(x,u)
        state = self.context.get_discrete_state()
        self.system.CalcDiscreteVariableUpdates(self.context, state)
        x_next = state.get_vector().value().flatten()

        return x_next

    def _calc_dynamics_partials(self, x, u):
        """
        Compute dynamics partials 

            x_next = f(x,u)
            fx = partial f(x,u) / partial x
            fu = partial f(x,u) / partial u
        """
        # Create autodiff versions of x and u
        xu = np.hstack([x,u])
        xu_ad = InitializeAutoDiff(xu)
        x_ad = xu_ad[:self.n]
        u_ad = xu_ad[self.n:]

        # Set input and state variables in our stored model accordingly
        self.context_ad.SetDiscreteState(x_ad)
        self.input_port_ad.FixValue(self.context_ad, u_ad)

        # Compute the forward dynamics x_next = f(x,u)
        state = self.context_ad.get_discrete_state()
        self.system_ad.CalcDiscreteVariableUpdates(self.context_ad, state)
        x_next = state.get_vector().CopyToVector()
       
        # Compute partial derivatives
        G = ExtractGradient(x_next)
        fx = G[:,:self.n]
        fu = G[:,self.n:]

        # Return exact value
        x_next = ExtractValue(x_next).flatten()

        return x_next, fx, fu

    def _forward_pass(self):
        """
        Simulate the system forward in time.

        Updates:
            u_bar:  The current best-guess of optimal u
            x_bar:  The current best-guess of optimal x
            fx:     Dynamics gradient w.r.t. x
            fu:     Dynamics gradient w.r.t. u

        Returns:
            L:          Total cost associated with this pass
        """
        L = 0
        x = np.zeros((self.n,self.N))
        u = np.zeros((self.m,self.N-1))

        x[:,0] = self.x0
        for t in range(0,self.N-1):
            # Update control input
            u[:,t] = self.u_bar[:,t]

            # Compute next state and dynamics derivatives
            x[:,t+1], self.fx[:,:,t], self.fu[:,:,t] = \
                    self._calc_dynamics_partials(x[:,t], u[:,t])
            
            # Update the current cost
            L += (x[:,t]-self.x_nom).T@self.Q@(x[:,t]-self.x_nom) + u[:,t].T@self.R@u[:,t]
        L += (x[:,-1]-self.x_nom).T@self.Qf@(x[:,-1]-self.x_nom)

        # Update stored values
        self.u_bar = u
        self.x_bar = x

        return L
    
    def _backward_pass(self):
        """
        Update the estimated costate trajectory by simuating backwards
        in time, where

            lambda_t = l_x + f_x^T lambda_{t+1}
            lambda_T = lf_x

        Updates:
            costate:    the costate (lambda) trajectory
        """
        # Gradient of terminal cost is final costate value
        lf_x, lf_xx = self._terminal_cost_partials(self.x_bar[:,-1])
        self.costate[:,-1] = lf_x

        # Do the backwards sweep
        for t in np.arange(self.N-2,-1,-1):
            lx, lu, lxx, luu, lux = \
                self._running_cost_partials(self.x_bar[:,t], self.u_bar[:,t])
            fx = self.fx[:,:,t]

            self.costate[:,t] = lx + fx.T@self.costate[:,t+1]

    def _update_control(self):
        """
        Update the best guess of the optimal control based on the 
        latest costate estimate:

            ubar_t = argmin_u H(x_t, u, lambda_{t+1})

        Updates:    
            u_bar:  The current control input guess
        """
        for t in range(self.N-1):
            lmbda = self.costate[:,t+1]
            fu = self.fu[:,:,t]

            u_star = -1/2 * np.linalg.inv(self.R) @ fu.T @ lmbda

            # Just move slightly in the direction of the new optimum
            alpha = 1e-5
            self.u_bar[:,t] = alpha*u_star + (1-alpha)*self.u_bar[:,t]

    def Solve(self):
        """
        Solve the optimization problem and return the (locally) optimal
        state and input trajectories. 

        Return:
            x:              (n,N) numpy array containing optimal state trajectory
            u:              (m,N-1) numpy array containing optimal control tape
            solve_time:     Total solve time in seconds
            optimal_cost:   Total cost associated with the (locally) optimal solution
        """
        # Store total cost and improvement in cost
        L = np.inf
        improvement = np.inf

        # Print labels for debug info
        print("---------------------------------------------------------------------------------")
        print("|    iter    |    cost    |    eps    |    ls    |    iter time    |    time    |")
        print("---------------------------------------------------------------------------------")

        # iteration counter
        i = 1
        st = time.time()
        while i < 100:
            st_iter = time.time()

            # Forward simulation to define state trajectory
            L_new = self._forward_pass()
            eps = np.inf      # DEBUG
            ls_iters = np.inf

            # Backward pass to define costate trajectory
            self._backward_pass()

            # Update control inputs
            self._update_control()

            iter_time = time.time() - st_iter
            total_time = time.time() - st

            print(f"{i:^14}{L_new:11.4f}  {eps:^12.4f}{ls_iters:^11}     {iter_time:1.5f}          {total_time:4.2f}")

            improvement = L - L_new
            L = L_new
            i += 1

        return self.x_bar, self.u_bar, total_time, L

