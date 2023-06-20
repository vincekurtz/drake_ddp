##
#
# A simple implementation of iterative LQR (iLQR) for discrete-time systems in Drake.
#
##

from pydrake.all import *
import time
import numpy as np

class IterativeLinearQuadraticRegulator():
    """
    Set up and solve a trajectory optimization problem of the form

        min_{u} sum{ (x-x_nom)'Q(x-x_nom) + u'Ru } + (x-x_nom)'Qf(x-x_nom)
        s.t.    x_{t+1} = f(x_t, u_t)

    using iLQR.
    """
    def __init__(self, system, num_timesteps, 
            input_port_index=0, delta=1e-2, beta=0.95, gamma=0.0):
        """
        Args:
            system:             Drake System describing the discrete-time dynamics
                                 x_{t+1} = f(x_t,u_t). Must be discrete-time.
            num_timesteps:      Number of timesteps to consider in the optimization.
            input_port_index:   InputPortIndex for the control input u_t. Default is to
                                 use the first port. 
            delta:              Termination criterion - the algorithm ends when the improvement
                                 in the total cost is less than delta. 
            beta:               Linesearch parameter in (0,1). Higher values lead to smaller
                                 linesearch steps. 
            gamma:              Linesearch parameter in [0,1). Higher values mean linesearch
                                 is performed more often in hopes of larger cost reductions.
        """
        assert system.IsDifferenceEquationSystem()[0],  "must be a discrete-time system"

        # float-type copy of the system and context for linesearch.
        # Computations using this system are fast but don't support gradients
        self.system = system
        self.context = self.system.CreateDefaultContext()
        self.input_port = self.system.get_input_port(input_port_index)

        # Autodiff copy of the system for computing dynamics gradients
        self.system_ad = system.ToAutoDiffXd()
        self.context_ad = self.system_ad.CreateDefaultContext()
        self.input_port_ad = self.system_ad.get_input_port(input_port_index)
       
        # Set some parameters
        self.N = num_timesteps
        self.delta = delta
        self.beta = beta
        self.gamma = gamma

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

        # Local feedback gains u = u_bar - eps*kappa_t - K_t*(x-x_bar)
        self.kappa = np.zeros((self.m,self.N-1))
        self.K = np.zeros((self.m,self.n,self.N-1))

        # Coefficents Qu'*Quu^{-1}*Qu for computing the expected 
        # reduction in cost dV = sum_t eps*(1-eps/2)*Qu'*Quu^{-1}*Qu
        self.dV_coeff = np.zeros(self.N-1)

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
        self.system.CalcForcedDiscreteVariableUpdate(self.context, state)
        x_next = state.get_vector().value().flatten()

        return x_next

    def _calc_dynamics_partials(self, x, u):
        """
        Given a system state (x) and a control input (u),
        compute the first-order partial derivitives of the dynamics

            x_next = f(x,u)
            fx = partial f(x,u) / partial x
            fu = partial f(x,u) / partial u
        
        Args:   
            x:  An (n,) numpy array representing the state
            u:  An (m,) numpy array representing the control input

        Returns:
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
        self.context_ad.SetDiscreteState(x_ad)
        self.input_port_ad.FixValue(self.context_ad, u_ad)

        # Compute the forward dynamics x_next = f(x,u)
        state = self.context_ad.get_discrete_state()
        self.system_ad.CalcForcedDiscreteVariableUpdate(self.context_ad, state)
        x_next = state.get_vector().CopyToVector()
       
        # Compute partial derivatives
        G = ExtractGradient(x_next)
        fx = G[:,:self.n]
        fu = G[:,self.n:]

        return (fx, fu)

    def _linesearch(self, L_last):
        """
        Determine a value of eps in (0,1] that results in a suitably
        reduced cost, based on forward simulations of the system. 

        This involves simulating the system according to the control law

            u = u_bar - eps*kappa - K*(x-x_bar).

        and reducing eps by a factor of beta until the improvement in
        total cost is greater than gamma*(expected cost reduction)

        Args:
            L_last: Total cost from the last iteration.

        Returns:
            eps:        Linesearch parameter
            x:          (n,N) numpy array of new states
            u:          (m,N-1) numpy array of new control inputs
            L:          Total cost/loss associated with the new trajectory
            n_iters:    Number of linesearch iterations taken

        Raises:
            RuntimeError: if eps has been reduced to <1e-8 and we still
                           haven't found a suitable parameter.
        """
        eps = 1.0
        n_iters = 0
        while eps >= 1e-8:
            n_iters += 1

            # Simulate system forward using the given eps value
            L = 0
            expected_improvement = 0
            x = np.zeros((self.n,self.N))
            u = np.zeros((self.m,self.N-1))

            x[:,0] = self.x0
            for t in range(0,self.N-1):
                u[:,t] = self.u_bar[:,t] - eps*self.kappa[:,t] - self.K[:,:,t]@(x[:,t] - self.x_bar[:,t])
                   
                try:
                    x[:,t+1] = self._calc_dynamics(x[:,t], u[:,t])
                except RuntimeError as e:
                    # If dynamics are infeasible, consider the loss to be infinite 
                    # and stop simulating. This will lead to a reduction in eps
                    print("Warning: encountered infeasible simulation in linesearch")
                    #print(e)
                    L = np.inf
                    break

                L += (x[:,t]-self.x_nom).T@self.Q@(x[:,t]-self.x_nom) + u[:,t].T@self.R@u[:,t]
                expected_improvement += -eps*(1-eps/2)*self.dV_coeff[t]
            L += (x[:,-1]-self.x_nom).T@self.Qf@(x[:,-1]-self.x_nom)

            # Chech whether the improvement is sufficient
            improvement = L_last - L
            if improvement > self.gamma*expected_improvement:
                return eps, x, u, L, n_iters

            # Otherwise reduce eps by a factor of beta
            eps *= self.beta

        raise RuntimeError("linesearch failed after %s iterations"%n_iters)
    
    def _forward_pass(self, L_last):
        """
        Simulate the system forward in time using the local feedback
        control law

            u = u_bar - eps*kappa - K*(x-x_bar).

        Performs a linesearch on eps to (approximately) determine the 
        largest value in (0,1] that results in a reduced cost. 

        Args:
            L_last: Total loss from last iteration, used for linesearch

        Updates:
            u_bar:  The current best-guess of optimal u
            x_bar:  The current best-guess of optimal x
            fx:     Dynamics gradient w.r.t. x
            fu:     Dynamics gradient w.r.t. u

        Returns:
            L:          Total cost associated with this iteration
            eps:        Linesearch parameter used
            ls_iters:   Number of linesearch iterations
        """
        # Do linesearch to determine eps
        eps, x, u, L, ls_iters = self._linesearch(L_last)

        # Compute the first-order dynamics derivatives
        for t in range(0,self.N-1):
            self.fx[:,:,t], self.fu[:,:,t] = self._calc_dynamics_partials(x[:,t], u[:,t])

        # Update stored values
        self.u_bar = u
        self.x_bar = x

        return L, eps, ls_iters
    
    def _backward_pass(self):
        """
        Compute a quadratic approximation of the optimal cost-to-go
        by simulating the system backward in time. Use this quadratic 
        approximation and a first-order approximation of the system 
        dynamics to compute the feedback controller

            u = u_bar - eps*kappa - K*(x-x_bar).

        Updates:
            kappa:      feedforward control term at each timestep
            K:          feedback control term at each timestep
            dV_coeff:   coefficients for expected change in cost
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

            # Derive cost reduction parameters
            self.dV_coeff[t] = Qu.T@Quu_inv@Qu

            # Update gradient and hessian of cost-to-go
            Vx = Qx - Qu.T@Quu_inv@Qux
            Vxx = Qxx - Qux.T@Quu_inv@Qux

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

        return self.x_bar, self.u_bar, total_time, L

    def SaveSolution(self, fname):
        """
        Save the stored solution, including target state x_bar
        nominal control input u_bar, feedback gains K, and timesteps
        t in the given file, where the feedback control

            u = u_bar - K*(x-x_bar)

        locally stabilizes the nominal trajectory.

        Args:
            fname:  npz file to save the data to.
        """
        dt = self.system.GetSubsystemByName("plant").time_step()
        T = (self.N-1)*dt
        t = np.arange(0,T,dt)

        x_bar = self.x_bar[:,:-1]  # remove last timestep
        u_bar = self.u_bar
        K = self.K

        np.savez(fname, t=t, x_bar=x_bar, u_bar=u_bar, K=K)
