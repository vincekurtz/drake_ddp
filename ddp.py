##
#
# A simple implementation of Differential Dynamic Programming (DDP)
#
##

from pydrake.all import *

class DifferentialDynamicProgramming():
    """
    Set up and solve a trajectory optimization problem of the form

        min_{u} sum{ (x-x_nom)'Q(x-x_nom) + u'Ru } + (x-x_nom)'Qf(x-x_nom)
        s.t.    x_{t+1} = f(x_t, u_t)

    using DDP.
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

        # Define default cost
        self.l = lambda x, u : 0
        self.lf = lambda xf : 0

        # Arrays to store best guess of control and state trajectory
        self.x_bar = np.zeros((self.n,self.N))
        self.u_bar = np.zeros((self.m,self.N-1))

        # List of approximate cost-to-go
        self.V = []

    def SetInitialState(self, x0):
        """
        Fix the initial condition for the optimization.

        Args:
            x0: Vector containing the initial state of the system
        """
        self.plant.SetPositionsAndVelocities(self.context, x0)

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
        self.U = u_guess

    def SetControlLimits(self, u_min, u_max):
        pass

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

        return (ExtractValue(x_next), fx, fu)

    def _forward_pass(self):
        """
        Simulate the system forward in time using the current
        """
        pass

    def _backward_pass(self):
        pass

    def _update_control(self):
        pass

    def Solve(self):
        """
        Solve the optimization problem and return the (locally) optimal
        state and input trajectories. 

        Return:
            x:  (n,N) numpy array containing optimal state trajectory
            u:  (m,N-1) numpy array containing optimal control tape
        """
        x = np.array([0,0,0,0])
        u = np.array([1])

        x_next, fx, fu = self._calc_dynamics(x, u)

        print(x_next)
        print(fx)
        print(fu)

