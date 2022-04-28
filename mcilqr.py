##
#
# Implementation of a stochastic optimal control algorithm based
# on a Monte-Carlo variant of iLQR.
#
##

from pydrake.all import *
import time
import numpy as np

class MonteCarloIterativeLQR():
    """
    Set up and solve a stochastic trajectory optimization problem of the form 

        min_{u} E[ sum{ (x-x_nom)''Q(x-x_nom) + u'Ru } + (x-x_nom)'Qf(x-x_nom) ]
        s.t.    x_{t+1} = f(x_t, u_t)
                x0 ~ N(mu, Sigma)

    using a monte-carlo variant of iLQR. The basic idea is to sample x0 from the
    initial distribution and then optimize over all samples simultaneously,
    where the same control tape is used for all of the samples. 
    """
    def __init__(self, system, num_timesteps, num_samples, seed=None, 
            input_port_index=0, delta=1e-2,beta=0.95, gamma=0.0):
        """
        Args:
            system:             Drake System describing the discrete-time dynamics
                                 x_{t+1} = f(x_t,u_t). Must be discrete-time.
            num_timesteps:      Number of timesteps to consider in the optimization.
            num_samples:        Number of samples to take from the initial
                                 distribution, to approximate the expected cost.
            seed:               Random seed to use for sampling from the initial
                                 distribution.
            input_port_index:   InputPortIndex for the control input u_t. Default is
                                 to use the first port. 
            delta:              Termination criterion - the algorithm ends when the 
                                 improvement in the total cost is less than delta. 
            beta:               Linesearch parameter in (0,1). Higher values lead 
                                 to smaller linesearch steps. 
            gamma:              Linesearch parameter in [0,1). Higher values mean 
                                 linesearch is performed more often in hopes of
                                 larger cost reductions.
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
        self.ns = num_samples
        self.delta = delta
        self.beta = beta
        self.gamma = gamma

        # Define state and input sizes
        self.nx = self.context.get_discrete_state_vector().size()
        self.n = self.nx*num_samples    # state is z = [x0,x1,...,x{ns}]
        self.m = self.input_port.size()

        # Initial and target states
        self.z0 = np.zeros(self.n)
        self.x_nom = np.zeros(self.nx)

        # Quadratic cost terms
        self.Q = np.eye(self.nx)
        self.R = np.eye(self.m)
        self.Qf = np.eye(self.nx)

        # Arrays to store best guess of control and state trajectory
        self.z_bar = np.zeros((self.n,self.N))
        self.u_bar = np.zeros((self.m,self.N-1))

        # Arrays to store dynamics gradients
        self.fz = np.zeros((self.n,self.n,self.N-1))
        self.fu = np.zeros((self.n,self.m,self.N-1))

        # Local feedback gains u = u_bar - eps*kappa_t - K_t*(z-z_bar)
        self.kappa = np.zeros((self.m,self.N-1))
        self.K = np.zeros((self.m,self.n,self.N-1))

        # Coefficents Qu'*Quu^{-1}*Qu for computing the expected 
        # reduction in cost dV = sum_t eps*(1-eps/2)*Qu'*Quu^{-1}*Qu
        self.dV_coeff = np.zeros(self.N-1)

        # Set seed for pseudorandom number generation
        np.random.seed(seed)

    def SetInitialDistribution(self, mu, Sigma):
        """
        Define the distribution of initial states,

            x0 ~ N(mu, Sigma).

        This method takes num_samples samples from this distribution to
        use for the optimization.
        """
        self.z0 = np.random.multivariate_normal(mu, Sigma,
                size=self.ns).flatten()
    
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

    def _running_cost_partials(self, z, u):
        """
        Return the partial derivatives of the (quadratic) running cost

            l(z,u) = 1/ns * sum_i[ x'Qx + u'Ru ]

        for the given state and input values.

        Args:
            z:  numpy array representing state samples z = [x0,x1,...]
            u:  numpy array representing control

        Returns:
            lz:     1st order partial w.r.t. z
            lu:     1st order partial w.r.t. u
            lzz:    2nd order partial w.r.t. z
            luu:    2nd order partial w.r.t. u
            luz:    2nd order partial w.r.t. u and z
        """
        lz = 2*self.Q@z - 2*self.x_nom.T@self.Q
        lu = 2*self.R@u
        lzz = 2*self.Q
        luu = 2*self.R
        luz = np.zeros((self.m,self.n))

        return (lz, lu, lzz, luu, luz)

    def _terminal_cost_partials(self, z):
        """
        Return the partial derivatives of the (quadratic) terminal cost

            lf(z) = 1/ns * sum_i[ x'Qfx ]

        for the given state values. 

        Args:
            z: numpy array representing state samples z = [x0,x1,...]

        Returns:
            lf_z:   gradient of terminal cost
            lf_zz:  hessian of terminal cost
        """
        lf_z = 2*self.Qf@z - 2*self.x_nom.T@self.Qf
        lf_zz = 2*self.Qf

        return (lf_z, lf_zz)
    
    def _calc_dynamics(self, z, u):
        """
        Given a set of system state samples (z) and a control input (u),
        compute the next state 

            z_next = f(z,u)

        Args:   
            z:  An (n,) numpy array representing the state samples
                 z = [x0,x1,...]
            u:  An (m,) numpy array representing the control input

        Returns:
            z_next: An (n,) numpy array representing the next state samples
        """
        x = z  # DEBUG: this should eventually go in a for loop

        # Set input and state variables in our stored model accordingly
        self.context.SetDiscreteState(x)
        self.input_port.FixValue(self.context, u)

        # Compute the forward dynamics x_next = f(x,u)
        state = self.context.get_discrete_state()
        self.system.CalcDiscreteVariableUpdates(self.context, state)
        x_next = state.get_vector().value().flatten()

        z_next = x_next

        return z_next

    def _calc_dynamics_partials(self, z, u):
        """
        Compute dynamics partials 

            x_next = f(z,u)
            fx = partial f(z,u) / partial z
            fu = partial f(z,u) / partial u

        using automatic differentiation.
        """
        x = z

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

        fz = fx 
        
        return (fz, fu)

    def _linesearch(self, L_last):
        """
        Determine a value of eps in (0,1] that results in a suitably
        reduced cost, based on forward simulations of the system. 

        This involves simulating the system according to the control law

            u = u_bar - eps*kappa - K*(z-z_bar).

        and reducing eps by a factor of beta until the improvement in
        total cost is greater than gamma*(expected cost reduction)

        Args:
            L_last: Total cost from the last iteration.

        Returns:
            eps:        Linesearch parameter
            z:          (n,N) numpy array of new state samples
            u:          (m,N-1) numpy array of new control inputs
            L:          Total cost/loss associated with the new trajectory
            n_iters:    Number of linesearch iterations taken
        """
        eps = 1.0
        n_iters = 0
        while eps >= 1e-8:
            n_iters += 1

            # Simulate system forward using the given eps value
            L = 0
            expected_improvement = 0
            z = np.zeros((self.n,self.N))
            u = np.zeros((self.m,self.N-1))

            z[:,0] = self.z0
            for t in range(0,self.N-1):
                u[:,t] = self.u_bar[:,t] - eps*self.kappa[:,t] - self.K[:,:,t]@(z[:,t] - self.z_bar[:,t])

                try:
                    z[:,t+1] = self._calc_dynamics(z[:,t], u[:,t])
                except RuntimeError as e:
                    # If dynamics are infeasible, consider the loss to be infinite 
                    # and stop simulating. This will lead to a reduction in eps
                    print("Warning: encountered infeasible simulation in linesearch")
                    L = np.inf
                    break

                L += (z[:,t]-self.x_nom).T@self.Q@(z[:,t]-self.x_nom) + u[:,t].T@self.R@u[:,t]
                expected_improvement += -eps*(1-eps/2)*self.dV_coeff[t]
            L += (z[:,-1]-self.x_nom).T@self.Qf@(z[:,-1]-self.x_nom)

            # Chech whether the improvement is sufficient
            improvement = L_last - L
            if improvement > self.gamma*expected_improvement:
                return eps, z, u, L, n_iters

            # Otherwise reduce eps by a factor of beta
            eps *= self.beta

        print(f"Warning: terminating linesearch after {n_iters} iterations")
        return eps, z, u, L, n_iters
    
    def _forward_pass(self, L_last):
        """
        Simulate the system forward in time using the local feedback
        control law

            u = u_bar - eps*kappa - K*(z-z_bar).

        Performs a linesearch on eps to (approximately) determine the 
        largest value in (0,1] that results in a reduced cost. 

        Args:
            L_last: Total loss from last iteration, used for linesearch

        Updates:
            u_bar:  The current best-guess of optimal u
            z_bar:  The current best-guess of optimal z
            fz:     Dynamics gradient w.r.t. z
            fu:     Dynamics gradient w.r.t. u

        Returns:
            L:          Total cost associated with this iteration
            eps:        Linesearch parameter used
            ls_iters:   Number of linesearch iterations
        """
        # Do linesearch to determine eps
        eps, z, u, L, ls_iters = self._linesearch(L_last)

        # Compute the first-order dynamics derivatives
        for t in range(0,self.N-1):
            self.fz[:,:,t], self.fu[:,:,t] = self._calc_dynamics_partials(z[:,t], u[:,t])

        # Update stored values
        self.u_bar = u
        self.z_bar = z

        return L, eps, ls_iters
    
    def _backward_pass(self):
        """
        Compute a quadratic approximation of the optimal cost-to-go
        by simulating the system backward in time. Use this quadratic 
        approximation and a first-order approximation of the system 
        dynamics to compute the feedback controller

            u = u_bar - eps*kappa - K*(z-z_bar).

        Updates:
            kappa:      feedforward control term at each timestep
            K:          feedback control term at each timestep
            dV_coeff:   coefficients for expected change in cost
        """
        # Store gradient and hessian of cost-to-go
        Vz, Vzz = self._terminal_cost_partials(self.z_bar[:,-1])

        # Do the backwards sweep
        for t in np.arange(self.N-2,-1,-1):
            z = self.z_bar[:,t]
            u = self.u_bar[:,t]

            # Get second(/first) order approximation of cost(/dynamics)
            lz, lu, lzz, luu, luz = self._running_cost_partials(z,u)
            fz = self.fz[:,:,t]
            fu = self.fu[:,:,t]

            # Construct second-order approximation of cost-to-go
            Qz = lz + fz.T@Vz
            Qu = lu + fu.T@Vz
            Qzz = lzz + fz.T@Vzz@fz
            Quu = luu + fu.T@Vzz@fu
            Quu_inv = np.linalg.inv(Quu)
            Quz = luz + fu.T@Vzz@fz

            # Derive controller parameters
            self.kappa[:,t] = Quu_inv@Qu
            self.K[:,:,t] = Quu_inv@Quz

            # Derive cost reduction parameters
            self.dV_coeff[t] = Qu.T@Quu_inv@Qu

            # Update gradient and hessian of cost-to-go
            Vz = Qz - Qu.T@Quu_inv@Quz
            Vzz = Qzz - Quz.T@Quu_inv@Quz

    def Solve(self):
        """
        Solve the optimization problem and return the (locally) optimal
        state and input trajectories. 

        Return:
            z:              (n,N) numpy array containing optimal state
                             trajectory samples
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
        while improvement > self.delta:
            st_iter = time.time()

            L_new, eps, ls_iters = self._forward_pass(L)
            self._backward_pass()

            iter_time = time.time() - st_iter
            total_time = time.time() - st

            print(f"{i:^14}{L_new:11.4f}  {eps:^12.4f}{ls_iters:^11}     {iter_time:1.5f}          {total_time:4.2f}")

            improvement = L - L_new
            L = L_new
            i += 1

        return self.z_bar, self.u_bar, total_time, L

    def SaveSolution(self, fname):
        """
        Save the stored solution, including target state samples z_bar
        nominal control input u_bar, feedback gains K, and timesteps
        t in the given file, where the feedback control

            u = u_bar - K*(z-z_bar)

        locally stabilizes the nominal trajectory.

        Args:
            fname:  npz file to save the data to.
        """
        dt = self.system.GetSubsystemByName("plant").time_step()
        T = (self.N-1)*dt
        t = np.arange(0,T,dt)

        z_bar = self.z_bar[:,:-1]  # remove last timestep
        u_bar = self.u_bar
        K = self.K

        np.savez(fname, t=t, z_bar=z_bar, u_bar=u_bar, K=K)
