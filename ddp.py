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
            plant:          MultibodyPlant describing the discrete-time dynamics
                            x_{t+1} = f(x_t,u_t)
            num_timesteps:  Number of timesteps to consider in the optimization
            eps:            Line search parameter in (0,1)
        """
        self.plant = plant
        self.context = plant.CreateDefaultContext()

        self.n = self.plant.num_multibody_states()
        self.m = self.plant.num_actuators()
        
        self.N = num_timesteps
        self.eps = eps

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

    def Solve(self):
        """
        Solve the optimization problem and return the (locally) optimal
        state and input trajectories. 

        Return:
            x:  (n,N) numpy array containing optimal state trajectory
            u:  (m,N-1) numpy array containing optimal control tape
        """

        pass
