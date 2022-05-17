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

        # Arrays to store current state, costate, and control extimates
        self.x = np.zeros((self.n,self.N))
        self.costate = np.zeros((self.n, self.N))
        self.u = np.zeros((self.m,self.N-1))
        
        # Arrays to store dynamics gradients
        self.fx = np.zeros((self.n,self.n,self.N-1))
        self.fu = np.zeros((self.n,self.m,self.N-1))

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
    
    def SetInitialGuess(self, x_guess, u_guess, lambda_guess):
        """
        Set the initial guess of state, control, and costate.

        Args:
            x_guess:      (n,N) numpy array containing x at each timestep
            u_guess:      (m,N-1) numpy array containing u at each timestep
            lambda_guess: (n,N)
        """
        assert u_guess.shape == (self.m, self.N-1)
        assert x_guess.shape == (self.n, self.N)
        assert lambda_guess.shape == (self.n, self.N)

        self.u = u_guess
        self.x = x_guess
        self.costate = lambda_guess  

        # TODO: consider setting costate guess from backwards pass

    def _calc_total_cost(self):
        """
        Compute the total cost associated with the current guess.
        """
        L = 0
        for t in range(self.N-1):
            x_err = self.x[:,t] - self.x_nom
            u = self.u[:,t]
            L += x_err.T@self.Q@x_err + u.T@self.R@u

        x_err = self.x[:,-1] - self.x_nom
        L += x_err.T@self.Qf@x_err

        return L

    def _calc_feasibility_gap(self):
        """
        Compute a measure of the dynamics error

            x_{t+1} - f(x_t, u_t)

        for the current guess. 
        """
        return np.inf

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

        return fx, fu

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
        # Print labels for debug info
        print("---------------------------------------------------------------------------------")
        print("|    iter    |    cost    |    gap    |    __    |    iter time    |    time    |")
        print("---------------------------------------------------------------------------------")

        # iteration counter
        i = 1
        st = time.time()
        while i <= 50:
            st_iter = time.time()

            # Stacking all constraints as g(y) = 0
            # where y = [x;l;u]. J(y) is the associated constraint jacobian.
            n_cons = 2*self.N*self.n + (self.N-1)*self.m
            J = np.zeros((n_cons, n_cons))
            g = np.zeros(n_cons)
           
            # Some convienient indeces for dealing with y = [x;l;u]
            x0_idx = 0
            l0_idx = self.N*self.n
            u0_idx = 2*self.N*self.n
            
            # Initial constraint
            g[0:self.n] = self.x[:,0] - self.x0
            J[0:self.n,0:self.n] = np.eye(self.n)

            # Dynamics constraints
            for t in range(self.N-1):
                fx, fu = self._calc_dynamics_partials(self.x[:,t], self.u[:,t])
                x_next = self._calc_dynamics(self.x[:,t], self.u[:,t])

                # Get some useful indices
                xt_idx = slice(x0_idx + t*self.n, x0_idx + (t+1)*self.n)
                x_next_idx = slice(x0_idx + (t+1)*self.n, x0_idx + (t+2)*self.n)
                lt_idx = slice(l0_idx + t*self.n, l0_idx + (t+1)*self.n)
                l_next_idx = slice(l0_idx + (t+1)*self.n, l0_idx + (t+2)*self.n)
                ut_idx = slice(u0_idx + t*self.m, u0_idx + (t+1)*self.m)

                # Forward dynamics constraints
                fdyn_idx = slice((t+1)*self.n, (t+2)*self.n)
                J[fdyn_idx, x_next_idx] = np.eye(self.n)
                J[fdyn_idx, xt_idx] = -fx
                J[fdyn_idx, ut_idx] = -fu

                g[fdyn_idx] = self.x[:,t+1] - x_next

                # Backwards dynamics constraints
                bdyn_idx = slice(self.N*self.n + t*self.n, self.N*self.n +
                        (t+1)*self.n)
                J[bdyn_idx, xt_idx] = -2*self.Q
                J[bdyn_idx, lt_idx] = np.eye(self.n)
                J[bdyn_idx, l_next_idx] = -fx.T

                g[bdyn_idx] = self.costate[:,t] - \
                        2*self.Q@(self.x[:,t]-self.x_nom) - \
                            fx.T@self.costate[:,t+1]

                # Optimal control constraints
                ctrl_idx = slice(2*self.N*self.n + t*self.m, 2*self.N*self.n +
                        (t+1)*self.m)
                J[ctrl_idx, ut_idx] = 2*self.R
                J[ctrl_idx, l_next_idx] = fu.T

                g[ctrl_idx] = 2*self.R@self.u[:,t] + fu.T@self.costate[:,t+1]

            # Costate boundary condition
            t = self.N-1
            xt_idx = slice(x0_idx + t*self.n, x0_idx + (t+1)*self.n)
            lt_idx = slice(l0_idx + t*self.n, l0_idx + (t+1)*self.n)
            bnd_idx = slice(2*self.N*self.n-self.n, 2*self.N*self.n)
            J[bnd_idx, lt_idx] = np.eye(self.n)
            J[bnd_idx, xt_idx] = -2*self.Qf

            g[bnd_idx] = self.costate[:,t] - 2*self.Qf@(self.x[:,t] - self.x_nom)

            y0 = np.hstack([
                    self.x.T.flatten(), 
                    self.costate.T.flatten(),
                    self.u.T.flatten()])
            
            # Solve the newton system
            alpha = 0.9
            dy = np.linalg.solve(J, -g)
            y = y0 + alpha*dy

            # Extract the solution
            self.x = y[x0_idx:l0_idx].reshape(self.N,self.n).T
            self.costate = y[l0_idx:u0_idx].reshape(self.N,self.n).T
            self.u = y[u0_idx:].reshape(self.N-1,self.m).T

            # Compute some stats
            L = self._calc_total_cost()
            gap = self._calc_feasibility_gap()
            placeholder = np.inf

            iter_time = time.time() - st_iter
            total_time = time.time() - st

            print(f"{i:^14}{L:11.4f}  {gap:^12.4f}{placeholder:^11}     {iter_time:1.5f}          {total_time:4.2f}")

            i += 1

        return self.x, self.u, total_time, L

