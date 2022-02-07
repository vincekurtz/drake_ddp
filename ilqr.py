##
#
# A simple implementation of iterative LQR (iLQR) for Drake MultibodyPlants.
#
##

from pydrake.all import *
import time

class IterativeLinearQuadraticRegulator():
    """
    Set up and solve a trajectory optimization problem of the form

        min_{u} sum{ (x-x_nom)'Q(x-x_nom) + u'Ru } + (x-x_nom)'Qf(x-x_nom)
        s.t.    x_{t+1} = f(x_t, u_t)

    using iLQR.
    """
    def __init__(self, plant, num_timesteps, eps=0.5):
        """
        Args:
            plant:          Drake MultibodyPlant describing the discrete-time dynamics
                             x_{t+1} = f(x_t,u_t)
            num_timesteps:  Number of timesteps to consider in the optimization
            eps:            Line search parameter in (0,1)
        """
        assert plant.IsDifferenceEquationSystem()[0],  "must be a discrete-time system"

        self.plant = plant.ToAutoDiffXd()   # convert to autodiff
        self.context = self.plant.CreateDefaultContext()

        self.n = self.plant.num_multibody_states()
        self.m = self.plant.num_actuators()
        
        self.N = num_timesteps
        self.eps = eps

        # Initial and target states
        self.x0 = None
        self.x_xom = None

        # Quadratic cost terms
        self.Q = None
        self.R = None
        self.Qf = None

        # Arrays to store best guess of control and state trajectory
        self.x_bar = np.zeros((self.n,self.N))
        self.u_bar = np.zeros((self.m,self.N-1))

        # Arrays to store dynamics gradients
        self.fx = None
        self.fu = None

        # Local feedback gains u = u_bar - eps*kappa_t - K_t*(x-x_bar)
        self.kappa = np.zeros((self.m,self.N-1))
        self.K = np.zeros((self.m,self.n,self.N-1))

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

    def SetControlLimits(self, u_min, u_max):
        pass

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

        along with the partial derivitives

            fx = partial f(x,u) / partial x
            fu = partial f(x,u) / partial u

        Args:   
            x:  An (n,) numpy array representing the state
            u:  An (m,) numpy array representing the control input

        Returns:
            x_next: An (n,) numpy array representing the next state
            fx:     A (n,n) numpy array representing the partial derivative 
                    of f with respect to x.
            fu:     A (n,m) numpy array representing the partial derivative 
                    of f with respect to u.
        """
        # Create autodiff versions of x and u
        xu = np.hstack([x,u])
        xu_ad = InitializeAutoDiff(xu)
        x_ad = xu_ad[:self.n]
        u_ad = xu_ad[self.n:]

        # Set input and state variables in our stored model accordingly
        self.plant.SetPositionsAndVelocities(self.context, x_ad)
        self.plant.get_actuation_input_port().FixValue(self.context, u_ad)

        # Compute the forward dynamics x_next = f(x,u)
        state = self.plant.AllocateDiscreteVariables()
        self.plant.CalcDiscreteVariableUpdates(self.context, state)
        x_next = state.get_vector().CopyToVector()
       
        # Compute partial derivatives
        G = ExtractGradient(x_next)
        fx = G[:,:self.n]
        fu = G[:,self.n:]

        return (ExtractValue(x_next).flatten(), fx, fu)

    def _forward_pass(self):
        """
        Simulate the system forward in time using the local feedback
        control law

            u = u_bar - eps*kappa - K*(x-x_bar).

        Updates:
            u_bar:  The current best-guess of optimal u
            x_bar:  The current best-guess of optimal x
            fx:     Dynamics gradient w.r.t. x
            fu:     Dynamics gradient w.r.t. u
        """
        x = np.zeros((self.n,self.N))
        u = np.zeros((self.m,self.N-1))
        fx = np.zeros((self.n,self.n,self.N-1))
        fu = np.zeros((self.n,self.m,self.N-1))

        # Set initial state
        x[:,0] = self.x0

        # simulate forward
        for t in range(0,self.N-1):
            u[:,t] = self.u_bar[:,t] - self.eps*self.kappa[:,t] - self.K[:,:,t]@(x[:,t] - self.x_bar[:,t])
            x[:,t+1], fx[:,:,t], fu[:,:,t] = self._calc_dynamics(x[:,t], u[:,t])

        # TODO: line search

        # Update stored values
        self.u_bar = u
        self.x_bar = x
        self.fx = fx
        self.fu = fu

    def _backward_pass(self):
        """
        Compute a quadratic approximation of the optimal cost-to-go
        by simulating the system backward in time. Use this quadratic 
        approximation and a first-order approximation of the system 
        dynamics to compute the feedback controller

            u = u_bar - eps*kappa - K*(x-x_bar).

        Updates:
            kappa:  feedforward control term at each timestep
            K:      feedback control term at each timestep
        """
        # Store gradient and hessian of cost-to-go
        Vx, Vxx = self._terminal_cost_partials(self.x_bar[:,-1])

        # Do the backwards sweep
        for t in np.arange(self.N-2,-1,-1):
            x = self.x_bar[:,t]
            u = self.u_bar[:,t]

            # Get second(/first) order approximation of cost(/dynamics)
            lx, lu, lxx, luu, lux = self._running_cost_partials(x,u)
            fx = self.fx[:,:,t]
            fu = self.fu[:,:,t]

            # Construct second-order approximation of cost-to-go
            Qx = lx + fx.T@Vx
            Qu = lu + fu.T@Vx
            Qxx = lxx + fx.T@Vxx@fx
            Quu = luu + fu.T@Vxx@fu
            Quu_inv = np.linalg.inv(Quu)
            Qux = lux + fu.T@Vxx@fx

            # Derive controller parameters
            self.kappa[:,t] = Quu_inv@Qu
            self.K[:,:,t] = Quu_inv@Qux

            # Update gradient and hessian of cost-to-go
            Vx = Qx - Qu.T@Quu_inv@Qux
            Vxx = Qxx - Qux.T@Quu_inv@Qux

    def Solve(self):
        """
        Solve the optimization problem and return the (locally) optimal
        state and input trajectories. 

        Return:
            x:  (n,N) numpy array containing optimal state trajectory
            u:  (m,N-1) numpy array containing optimal control tape
        """
        for i in range(20):
            print("Interation: ", i+1)
            self._forward_pass()
            
            # DEBUG: compute total cost from nominal trajectory
            L = 0
            for t in range(self.N-1):
                x = self.x_bar[:,t]
                u = self.u_bar[:,t]
                L += (x-self.x_nom).T@self.Q@(x-self.x_nom) + u.T@self.R@u
            x = self.x_bar[:,-1]
            L += (x.T-self.x_nom.T)@self.Qf@(x-self.x_nom)
            print("Cost: ", L)

            self._backward_pass()

        return self.x_bar, self.u_bar
