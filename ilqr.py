##
#
# A simple implementation of iterative LQR (iLQR) for discrete-time systems in Drake.
#
##

from pydrake.all import *
import time
import numpy as np
import matplotlib.pyplot as plt
import utils_derivs_interpolation

class IterativeLinearQuadraticRegulator():
    """
    Set up and solve a trajectory optimization problem of the form

        min_{u} sum{ (x-x_nom)'Q(x-x_nom) + u'Ru } + (x-x_nom)'Qf(x-x_nom)
        s.t.    x_{t+1} = f(x_t, u_t)

    using iLQR.
    """
    def __init__(self, system, num_timesteps, 
            input_port_index=0, delta=1e-2, beta=0.95, gamma=0.0, derivs_keypoint_method = None):
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

        # -------------------------------- Derivatives interpolation additions --------------------------------

        # Arrays for dynamics gradients with interpolation
        self.fx_baseline = np.zeros((self.n,self.n,self.N-1))
        self.fu_baseline = np.zeros((self.n,self.m,self.N-1))

        self.deriv_calculated_at_index = np.zeros(self.N-1, dtype=bool)
        self.time_getDerivs = 0
        self.percentage_derivs = 0
        self.time_backwardsPass = 0
        self.time_fp = 0

        # Total number of columns dynamics gradients over the trajectory (trajec length * dof) (we group dof columns into triplets)
        self.total_num_columns_derivs = self.N * self.n

        # If no derivs_interpolation specified - use the baseline case
        if derivs_keypoint_method is None:
            self.derivs_interpolation = utils_derivs_interpolation.derivs_interpolation('setInterval', 1, 0, 0, 0)
        else:
            self.derivs_interpolation = derivs_keypoint_method

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
        timeStart = time.time()
        eps, x, u, L, ls_iters = self._linesearch(L_last)
        timeEnd = time.time()
        self.time_fp = timeEnd - timeStart

        timeStart = time.time()
        self._get_derivatives(x, u)
        timeEnd = time.time()
        self.time_getDerivs = timeEnd - timeStart

        # Update stored values
        self.u_bar = u
        self.x_bar = x

        return L, eps, ls_iters

    def _get_derivatives(self, x, u):
        """
        Calculates the derivatives fx and fu over the entire trajectory. Depending on 
        the keypoint method specified, this function will calculate a set of keypoints.
        At these keypoints the derivatives will be calculated exactly using autodiff.
        In-between these keypoints this function will linearly interpolate approximations
        to the derivatives between the computed values. This is done to reduce the amount
        of time spent per iteration computing derivatives.

        Updates:
            fx:      dynamcis partial wrt state at each timestep
            fu:      dynamcis partial wrt control at each timestep
        """

        # Debug variable to plot some useful graphs if required, also calculates the 
        # derivatives at every time step and compute the error of the interpolation
        # for debugging
        DEBUG = False

        if(DEBUG and self.derivs_interpolation.keypoint_method == 'adaptiveJerk'):
            jerkProfile = self.calc_jerk_profile(x)
            plt.title("jerk profile")
            plt.plot(jerkProfile[:,0])
            plt.show()

        # Calculate keypoints over the trajectory
        keyPoints = []
        if(self.derivs_interpolation.keypoint_method == 'setInterval'):
            keyPoints = self.get_keypoints_set_interval()
        elif(self.derivs_interpolation.keypoint_method == 'adaptiveJerk'):
            keyPoints = self.get_keypoints_adaptive_jerk(x, u)
        elif(self.derivs_interpolation.keypoint_method == 'iterativeError'):
            keyPoints = self.get_keypoints_iterative_error(x, u)
            self.deriv_calculated_at_index = [False] * self.N
        else:
            raise Exception('unknown interpolation method')

        self.percentage_derivs = (len(keyPoints) / (self.N - 1)) * 100

        if(DEBUG):
            for t in range(self.N-1):
                self.fx_baseline[:,:,t], self.fu_baseline[:,:,t] = self._calc_dynamics_partials(x[:,t], u[:,t])

        # Calculate derivatives at keypoints
        if self.derivs_interpolation.keypoint_method != 'iterativeError':
            for t in range(len(keyPoints)):
                self.fx[:,:,keyPoints[t]], self.fu[:,:,keyPoints[t]] = self._calc_dynamics_partials(x[:,keyPoints[t]], u[:,keyPoints[t]])

        # Interpolate derivatives if required
        if not (self.derivs_interpolation.keypoint_method == 'setInterval' and self.derivs_interpolation.minN == 1):
            self.interpolate_derivatives(keyPoints)
        
        indexX = 0
        indexY = 3
        error_fx, error_fu = self.calc_error_of_interpolation()

        if(DEBUG):  
            print(f'error fx: {error_fx} error fu:  {error_fu}')
            plt.title(f"error fx row {indexX} col {indexY}")
            plt.plot(self.fx[indexX,indexY,:], label="interpolations")
            plt.plot(self.fx_baseline[indexX,indexY,:], label="baseline")
            plt.legend()
            plt.show()    

        # if(DEBUG):
        #     error_fx, error_fu = self.calc_error_of_interpolation()

        #     print(f'error fx: {error_fx} error fu:  {error_fu}')
        #     for i in range(self.n):
        #         for j in range(self.n):
        #             plt.title(f"error fx row {i} col {j}")
        #             plt.plot(self.fx[i,j,:], label="interpolations")
        #             plt.plot(self.fx_baseline[i,j,:], label="baseline")
        #             plt.legend()
        #             plt.show()

    def get_keypoints_set_interval(self):
        """
        Computes keypoints over the trajectory at set intervals as specified by the
        interpolation method

        Updates:
            N/A
        """

        keypoints = []

        keypoints = np.arange(0,self.N-1, self.derivs_interpolation.minN).astype(int)
        if keypoints[-1] != self.N-2:
            keypoints[-1] = self.N-2

        return keypoints

    def get_keypoints_adaptive_jerk(self, x, u):
        """
        Computes keypoints over the trajectory adaptively by looking at the jerk
        profile over the trajectory and changing the sample rate based on the jerk

        Updates:
            N/A
        """
        keypoints = []

        dof = int(self.n/2)

        jerk_profile = self.calc_jerk_profile(x)
        counter = 0
        keypoints.append(0)

        for t in range(len(jerk_profile)):
            counter += 1

            if counter >= self.derivs_interpolation.minN:
                for i in range(dof):
                    if jerk_profile[t, i] > self.derivs_interpolation.jerk_threshold:
                        keypoints.append(t)
                        counter = 0
                        break
            
            if counter >= self.derivs_interpolation.maxN:
                keypoints.append(t)
                counter = 0

            
        if keypoints[-1] != self.N-2:
            keypoints[-1] = self.N-2
                        

        return keypoints

    def calc_jerk_profile(self, x):
        """
        Calculates the jerk profile (derivative of acceleration) for each
        degree of freedom over the trajectory

        Updates:
            N/A
        """
        dof = int(self.n/2)
        jerk = np.zeros((self.N-3, dof))
        for i in range(dof):
            for t in range(self.N-3):
                acell1 = x[i + dof,t+2] - x[i + dof, t+1]
                acell2 = x[i + dof,t+1] - x[i + dof, t]

                jerk[t, i] = acell1 - acell2
                

        return jerk

    def get_keypoints_iterative_error(self, x, u):
        """
        Calculates keypoints at which to calcualte derivatives by checking
        middle of the inteprolation versus the real value. If the approximation 
        is valid, no further subdivisions are required. If the approximation is 
        bad, then further subdivisions are required.

        Updates:
            fx:      at certain timesteps
            fu:      at certain timesteps
        """
        keypoints = []
        binsComplete = False 
        
        start_index = 0
        end_index = self.N-2

        initial_index_tuple = utils_derivs_interpolation.index_tuple(start_index, end_index)
        list_indices_to_check = [initial_index_tuple]
        sub_list_with_midpoints = []

        while not binsComplete:
            sub_list_indices = []
            all_checks_passed = True
            for i in range(len(list_indices_to_check)):
                # print(f'checking indices {list_indices_to_check[i].start_index} to {list_indices_to_check[i].end_index}')

                approximation_good = self.check_one_matrix_error(list_indices_to_check[i], x, u)
                mid_index = int((list_indices_to_check[i].start_index + list_indices_to_check[i].end_index)/2)

                if not approximation_good:
                    sub_list_indices.append(utils_derivs_interpolation.index_tuple(list_indices_to_check[i].start_index, mid_index))
                    sub_list_indices.append(utils_derivs_interpolation.index_tuple(mid_index, list_indices_to_check[i].end_index))
                    all_checks_passed = False

                else:
                    sub_list_with_midpoints.append(list_indices_to_check[i].start_index)
                    sub_list_with_midpoints.append(mid_index)
                    sub_list_with_midpoints.append(list_indices_to_check[i].end_index)

            
            if(all_checks_passed):
                binsComplete = True

            list_indices_to_check = sub_list_indices
            sub_list_indices = []

        for i in range(self.N-1):
            if(self.deriv_calculated_at_index[i]):
                keypoints.append(i)


        return keypoints

    def check_one_matrix_error(self, indices, x, u):
        """
        Checks the mean sqaured sum error of two dynamics partials matrices
        If the error is above the set threshold, the approximation is bad 
        and false is returend. This leads to further subdivisions in the iterative
        error method.

        Updates:
            fx at certain timesteps
            fu at certain timesteps
        """
        approximation_good = True

        if(indices.end_index - indices.start_index <= self.derivs_interpolation.minN):
            return approximation_good

        start_index = indices.start_index
        mid_index = int((indices.start_index + indices.end_index)/2)
        end_index = indices.end_index

        if(not self.deriv_calculated_at_index[start_index]):
            # Calculate the graident matrices at this index
            self.fx[:,:,start_index], self.fu[:,:,start_index] = self._calc_dynamics_partials(x[:,start_index], u[:,start_index])
            self.deriv_calculated_at_index[start_index] = True

        if(not self.deriv_calculated_at_index[mid_index]):
            # Calculate the graident matrices at this index
            self.fx[:,:,mid_index], self.fu[:,:,mid_index] = self._calc_dynamics_partials(x[:,mid_index], u[:,mid_index])
            self.deriv_calculated_at_index[mid_index] = True

        if(not self.deriv_calculated_at_index[end_index]):
            # Calculate the graident matrices at this index
            self.fx[:,:,end_index], self.fu[:,:,end_index] = self._calc_dynamics_partials(x[:,end_index], u[:,end_index])
            self.deriv_calculated_at_index[end_index] = True

        #calculate mid index via interpolation
        fx_mid_lin = (self.fx[:,:,end_index] + self.fx[:,:,start_index] ) / 2


        sumSqDiff = 0
        for i in range(self.n):
            for j in range(self.n):
                sumSqDiff += (fx_mid_lin[i,j] - self.fx[i,j,mid_index])**2

        average_sq_diff = sumSqDiff / (2 * self.n)
        # print(f'average_sq_diff: {average_sq_diff}')

        if(average_sq_diff > self.derivs_interpolation.iterative_error_threshold):
            approximation_good = False


        return approximation_good


    def interpolate_derivatives(self, keyPoints):
        """
        Interpolate the dynamics partials (fx, fu) by linealry
        interpolating the calculated values at the set keypoints.

        Updates:
            fx:      dynamcis partial wrt state at each timestep
            fu:      dynamcis partial wrt control at each timestep
        """

        # Interpoalte whole matrices
        for i in range(len(keyPoints) - 1):
            startIndex = keyPoints[i]
            endIndex = keyPoints[i+1]

            startVals_fx = self.fx[:,:,startIndex]
            endVals_fx = self.fx[:,:,endIndex]
            startVals_fu = self.fu[:,:,startIndex]
            endVals_fu = self.fu[:,:,endIndex]

            diff_fx = endVals_fx - startVals_fx
            diff_fu = endVals_fu - startVals_fu

            for j in range(startIndex, endIndex):
                self.fx[:,:,j] = startVals_fx + (endVals_fx - startVals_fx) * (j - startIndex) / (endIndex - startIndex)
                self.fu[:,:,j] = startVals_fu + (endVals_fu - startVals_fu) * (j - startIndex) / (endIndex - startIndex)

    def calc_error_of_interpolation(self):
        error_fx = 0
        error_fu = 0

        for t in range(self.N-1):
            diff_fx = self.diff_between_matrices(self.fx[:,:,t], self.fx_baseline[:,:,t])
            diff_fu = self.diff_between_matrices(self.fu[:,:,t], self.fu_baseline[:,:,t])

            error_fx += diff_fx
            error_fu += diff_fu

        return error_fx, error_fu

    def diff_between_matrices(self, matrix1, matrix2):
        diff = 0

        for i in range(matrix1.shape[0]):
            for j in range(matrix1.shape[1]):
                diff += abs(matrix1[i,j] - matrix2[i,j])

        return diff
    
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
        print("----------------------------------------------------------------------------------------------------------------------------------")
        print("|    iter    |    cost    |    eps    |    ls    | derivs time | derivs '%'  | bp time  | fp time  |   iter time    |    time    |")
        print("----------------------------------------------------------------------------------------------------------------------------------")

        # iteration counter
        i = 1
        st = time.time()
        while improvement > self.delta:
            st_iter = time.time()

            L_new, eps, ls_iters = self._forward_pass(L)
            bp_time_start = time.time()
            self._backward_pass()
            bp_end_time = time.time()
            self.time_backwardsPass = bp_end_time - bp_time_start

            iter_time = time.time() - st_iter
            total_time = time.time() - st

            print(f"{i:^14}{L_new:11.4f}  {eps:^12.4f}{ls_iters:^11}   {self.time_getDerivs:1.5f}         {self.percentage_derivs:.1f}       {self.time_backwardsPass:1.5f}    {self.time_fp:1.5f}      {iter_time:1.5f}          {total_time:4.2f}")

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
