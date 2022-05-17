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

    def _calc_g(self):
        """
        Compute the value of the constraint function

            g(x,u,lambda) = 0

        which contains the PMP optimality conditions.
        """
        n = self.n
        m = self.m
        N = self.N
        g = np.zeros(2*N*n + (N-1)*m)

        # Initial condition x0 = x_init
        g[0:n] = self.x[:,0] - self.x0

        # Terminal costate condition lambda_T = lf_x(x_T)
        lf_x, _ = self._terminal_cost_partials(self.x[:,-1])
        g[n:2*n] = self.costate[:,-1] - lf_x

        # State dynamics x_{t+1} = f(x_t, u_t)
        for t in range(N-1):
            x_next = self.x[:,t+1]
            x_pred = self._calc_dynamics(self.x[:,t], self.u[:,t])
            g[(2+t)*n:(2+t+1)*n] = x_next - x_pred

        # Costate dynamics lambda_t = lx + fx'*lambda_{t+1}
        for t in range(N-1):
            lx, _,_,_,_ = self._running_cost_partials(self.x[:,t], self.u[:,t])
            fx = self.fx[:,:,t]

            lmbda = self.costate[:,t]
            lmbda_pred = lx + fx.T@self.costate[:,t+1]

            g[(2+N-1+t)*n:(2+N+t)*n] = lmbda - lmbda_pred

        # Optimal control conditions lu + fu'*lambda_{t+1} = 0
        start_idx = (2+2*(N-1))*n
        for t in range(N-1):
            _, lu, _,_,_ = self._running_cost_partials(self.x[:,t], self.u[:,t])
            fu = self.fu[:,:,t]
            lmbda = self.costate[:,t+1]

            g[start_idx+t*m:start_idx+(t+1)*m] = lu + fu.T@lmbda

        return g

    def _calc_grad_g(self):
        """
        Compute the gradient of the constraint function

            g(x,u,lambda),

        where g=0 enforces the PMP optimality conditions.
        """
        n = self.n
        m = self.m
        N = self.N
        size = 2*N*n + (N-1)*m  # number of variables and constraints
        grad = np.full((size, size), 0.0)

        # Initial condition x0 = x_init
        row_start = 0
        row_end = n

        grad[row_start:row_end,0:n] = np.eye(n)
        
        # Terminal costate condition lambda_T = lf_x(x_T)
        row_start = n
        row_end = 2*n

        _, lf_xx = self._terminal_cost_partials(self.x[:,-1])
        grad[row_start:row_end, 2*n:3*n] = -lf_xx

        grad[row_start:row_end, (2*N-1)*n:2*N*n] = np.eye(n)
        
        # State dynamics x_{t+1} = f(x_t, u_t)
        for t in range(N-1):
            row_start = (2+t)*n
            row_end = (2+t+1)*n

            fx = self.fx[:,:,t]
            fu = self.fu[:,:,t]

            grad[row_start:row_end, t*n:(t+1)*n] = -fx
            grad[row_start:row_end, (t+1)*n:(t+2)*n] = np.eye(n)

            col_start = (2+2*(N-1))*n+t*m
            col_end = (2+2*(N-1))*n+(t+1)*m
            grad[row_start:row_end, col_start:col_end] = -fu

        # Costate dynamics lambda_t = lx + fx'*lambda_{t+1}
        for t in range(N-1):
            row_start = (2+N-1+t)*n
            row_end = (2+N+t)*n

            _,_, lxx, _,_ = self._running_cost_partials(self.x[:,t],self.u[:,t])
            fx = self.fx[:,:,t]

            grad[row_start:row_end, t*n:(t+1)*n] = -lxx
            grad[row_start:row_end, (N+t)*n:(N+t+1)*n] = np.eye(n)
            grad[row_start:row_end, (N+t+1)*n:(N+t+2)*n] = -fx.T
        
        # Optimal control conditions lu + fu'*lambda_{t+1} = 0
        for t in range(N-1):
            row_start = (2+2*(N-1))*n+t*m
            row_end = (2+2*(N-1))*n+(t+1)*m

            _,_,_, luu ,_ = self._running_cost_partials(self.x[:,t],self.u[:,t])
            fu = self.fu[:,:,t]

            grad[row_start:row_end, (N+1+t)*n: (N+1+t+1)*n] = fu.T
            grad[row_start:row_end, row_start:row_end] = luu

        return grad

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
        while i <= 1:
            st_iter = time.time()

            # Update all dynamics gradients
            for t in range(self.N-1):
                self.fx[:,:,t], self.fu[:,:,t] = \
                        self._calc_dynamics_partials(self.x[:,t], self.u[:,t])

            # DEBUG
            mp = MathematicalProgram()
            x = mp.NewContinuousVariables(self.n, self.N, "x")
            l = mp.NewContinuousVariables(self.n, self.N, "lmbda")
            u = mp.NewContinuousVariables(self.m, self.N-1, "u")

            ## Initial
            #mp.AddConstraint(eq(0, x[:,0]-self.x0 ))
            #for t in range(self.N-1):
            #    A = self.fx[:,:,t]
            #    B = self.fu[:,:,t]

            #    # Forward (state) dynamics
            #    mp.AddConstraint(eq(
            #        0, x[:,t+1] - A@x[:,t] - B@u[:,t]
            #    ))

            #    # Backward (costate) dynamics
            #    mp.AddConstraint(eq(
            #        0, l[:,t] - 2*self.Q@(x[:,t]-self.x_nom) - A.T@l[:,t+1]
            #    ))

            #    # Optimal constrol condition
            #    mp.AddConstraint(eq(
            #        0, 2*self.R@u[:,t] + B.T@l[:,t+1]
            #    ))

            ## Costate boundary condition
            #mp.AddConstraint(eq(
            #    0, l[:,-1] - 2*self.Qf@(x[:,-1]-self.x_nom)
            #))

            # Stacking all constraints as G*[x;l;u] - h = 0
            n_cons = 2*self.N*self.n + (self.N-1)*self.m
            G = np.zeros((n_cons, n_cons))
            h = np.zeros(n_cons)
            y = np.hstack([x.T.flatten(), l.T.flatten(), u.T.flatten()])
           
            # Some convienient indeces for dealing with y = [x;l;u]
            x0_idx = 0
            l0_idx = self.N*self.n
            u0_idx = 2*self.N*self.n
            
            # Initial constraint
            h[0:self.n] = self.x0
            G[0:self.n,0:self.n] = np.eye(self.n)

            # Dynamics constraints
            for t in range(self.N-1):
                fx = self.fx[:,:,t]
                fu = self.fu[:,:,t]

                # Get some useful indices
                xt_idx = slice(x0_idx + t*self.n, x0_idx + (t+1)*self.n)
                x_next_idx = slice(x0_idx + (t+1)*self.n, x0_idx + (t+2)*self.n)
                lt_idx = slice(l0_idx + t*self.n, l0_idx + (t+1)*self.n)
                l_next_idx = slice(l0_idx + (t+1)*self.n, l0_idx + (t+2)*self.n)
                ut_idx = slice(u0_idx + t*self.m, u0_idx + (t+1)*self.m)

                # Forward dynamics constraints
                fdyn_idx = slice((t+1)*self.n, (t+2)*self.n)
                G[fdyn_idx, x_next_idx] = np.eye(self.n)
                G[fdyn_idx, xt_idx] = -fx
                G[fdyn_idx, ut_idx] = -fu

                # Backwards dynamics constraints
                bdyn_idx = slice(self.N*self.n + t*self.n, self.N*self.n +
                        (t+1)*self.n)
                G[bdyn_idx, xt_idx] = -2*self.Q
                G[bdyn_idx, lt_idx] = np.eye(self.n)
                G[bdyn_idx, l_next_idx] = -fx.T
                h[bdyn_idx] = -2*self.Q@self.x_nom

                # Optimal control constraints
                ctrl_idx = slice(2*self.N*self.n + t*self.m, 2*self.N*self.n +
                        (t+1)*self.m)
                G[ctrl_idx, ut_idx] = 2*self.R
                G[ctrl_idx, l_next_idx] = fu.T

            # Costate boundary condition
            t = self.N-1
            xt_idx = slice(x0_idx + t*self.n, x0_idx + (t+1)*self.n)
            lt_idx = slice(l0_idx + t*self.n, l0_idx + (t+1)*self.n)
            bnd_idx = slice(2*self.N*self.n-self.n, 2*self.N*self.n)
            G[bnd_idx, lt_idx] = np.eye(self.n)
            G[bnd_idx, xt_idx] = -2*self.Qf
            h[bnd_idx] = -2*self.Qf@self.x_nom

            mp.AddLinearEqualityConstraint(G, h, y)

            res = ClpSolver().Solve(mp)
            #res = Solve(mp)
            self.x = res.GetSolution(x)
            self.u = res.GetSolution(u)
            self.costate = res.GetSolution(l)

            ## Construct g(Y) and \grad g(Y)
            g = self._calc_g()
            #grad = self._calc_grad_g()

            ## Solve the newton system
            #grad += 0.01*np.eye(len(grad))
            #delta_Y = np.linalg.solve(grad, -g)

            ## Linesearch (?)
            #alpha = 0.1

            ## Update Y = [x, lambda, u]
            #Y = np.hstack([
            #        self.x.flatten(), 
            #        self.costate.flatten(),
            #        self.u.flatten()])
           
            #Y += alpha * delta_Y

            #self.x = Y[0:self.N*self.n].reshape(self.N,self.n).T
            #self.costate = Y[self.N*self.n:2*self.N*self.n].reshape(self.N,self.n).T
            #self.u = Y[2*self.N*self.n:].reshape(self.N-1,self.m).T

            # Compute some stats
            L = self._calc_total_cost()
            gap = self._calc_feasibility_gap()
            placeholder = np.inf

            iter_time = time.time() - st_iter
            total_time = time.time() - st

            print(f"{i:^14}{L:11.4f}  {gap:^12.4f}{placeholder:^11}     {iter_time:1.5f}          {total_time:4.2f}")

            i += 1

        return self.x, self.u, total_time, L

