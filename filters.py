
### Author Name: Manqing Mao
### GTID: mmao33
import numpy as np
import cv2


# Assignment code
class KalmanFilter(object):
    """A Kalman filter tracker"""

    def __init__(self, init_x, init_y, Q=0.1 * np.eye(4), R=0.1 * np.eye(2)):
        """Initializes the Kalman Filter

        Args:
            init_x (int or float): Initial x position.
            init_y (int or float): Initial y position.
            Q (numpy.array): Process noise array.
            R (numpy.array): Measurement noise array.
        """
        self.state = np.array([init_x, init_y, 0., 0.])  # state
        
        # State and Covariance prediction
        self.covariance = np.array([[1., 0., 0., 0.],
                                    [0., 1., 0., 0.],
                                    [0., 0., 0.1, 0.],
                                    [0., 0., 0., 0.1]])
        
        # State transition 4x4 matrix
        delta_t = 1.
        self.D_t = np.array([[1., 0., delta_t, 0.],
                             [0., 1., 0., delta_t],
                             [0., 0., 1., 0.],
                             [0., 0., 0., 1.]])

        # Measurement 2x4 matrix
        self.M_t = np.array([[1., 0., 0., 0.],
                             [0., 1., 0., 0.]])        
        # process noise 4x4 matrix
        self.sigma_dt = Q
        # measurement noise 2x2 matrix
        self.sigma_mt = R
        
    def predict(self):
        # update state matrix:
        self.state = np.dot(self.D_t, self.state)
        # update covariance matrix: 
        self.covariance = np.dot(self.D_t, np.dot(self.covariance, self.D_t.T)) + self.sigma_dt

    def correct(self, meas_x, meas_y):
        temp = np.dot(self.M_t, np.dot(self.covariance, self.M_t.T)) + self.sigma_mt
        K_t = np.dot(self.covariance, np.dot(self.M_t.T, np.linalg.inv(temp)))
        # observation
        Y_t = np.array([meas_x, meas_y])
        delta_MtXt = Y_t - np.dot(self.M_t, self.state)
        self.state = self.state + np.dot(K_t, delta_MtXt)
        # corrected covariance
        self.covariance = np.dot((np.eye(4) - np.dot(K_t, self.M_t)), self.covariance)

    def process(self, measurement_x, measurement_y):

        self.predict()
        self.correct(measurement_x, measurement_y)

        return self.state[0], self.state[1]


class ParticleFilter(object):
    """A particle filter tracker.

    Encapsulating state, initialization and update methods. Refer to
    the method run_particle_filter( ) in experiment.py to understand
    how this class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles.
                                        This should be a N x 2 array where
                                        N = self.num_particles. This component
                                        is used by the autograder so make sure
                                        you define it appropriately.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        - self.frame (numpy.array): Current image frame.

        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track,
                                    values in [0, 255].
            kwargs: keyword arguments needed by particle filter model:
                    - num_particles (int): number of particles.
                    - sigma_exp (float): sigma value used in the similarity
                                         measure.
                    - sigma_dyn (float): sigma value that can be used when
                                         adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y,
                                            width, and height values.
        """
        self.num_particles = kwargs.get('num_particles')  # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp')  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

        self.template = template
        self.frame = frame

        # Initialize your weights array. Read the docstring.
        self.weights = np.ones(self.num_particles) / self.num_particles  

        # Initialize your particles array. Read the docstring.
        # This Nx2 array represents the coordinates of N particles (random location)
        y_range, x_range = (frame.shape[0], frame.shape[1])
        particles_x = np.random.randint(x_range, size=self.num_particles)
        particles_y = np.random.randint(y_range, size=self.num_particles)
        particles_x = particles_x[:, np.newaxis]
        particles_y = particles_y[:, np.newaxis]
        self.particles = np.concatenate((particles_x, particles_y), axis=1)

        # USE IN PART3: class AppearanceModelPF(ParticleFilter):
        w_temp = self.template_rect['w']
        h_temp = self.template_rect['h']
        x_temp = self.template_rect['x']
        y_temp = self.template_rect['y']
        # Intial position: the center of the template in the first frame.
        self.best_particle = [x_temp + w_temp // 2, y_temp + h_temp // 2]

    def get_particles(self):
        """Returns the current particles state.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: particles data structure.
        """
        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """
        return self.weights

    def get_error_metric(self, template, frame_cutout):
        """Returns the error metric used based on the similarity measure.

        Returns:
            float: similarity value.
        """
        template_temp = np.copy(template)
        frame_cutout_temp = np.copy(frame_cutout)

        if template.ndim == 3:
            template_temp = cv2.cvtColor(template_temp, cv2.COLOR_BGR2GRAY)
            frame_cutout_temp = cv2.cvtColor(frame_cutout_temp, cv2.COLOR_BGR2GRAY)

        m = template_temp.shape[0]
        n = template_temp.shape[1]

        array_temp = np.square(np.float64(template_temp) - np.float64(frame_cutout_temp))
        MSE = np.sum(array_temp) / (m * n)
        similarity_val = np.exp(- MSE / (2 * np.square(self.sigma_exp)))
        
        return similarity_val

    def resample_particles(self):
        """Returns a new set of particles

        This method does not alter self.particles.

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.
        
        Returns:
            numpy.array: particles data structure.
        """
        particles_index = np.arange(self.num_particles)
        particlesID = np.random.choice(particles_index, self.num_particles, p=self.weights)

        resampled_particles = np.empty([self.num_particles, 2])

        for i in range(self.num_particles):
            resampled_particles[i, :] = self.particles[particlesID[i], :]

        return resampled_particles

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        temp_frame = np.copy(frame)

        w = self.template_rect['w']
        h = self.template_rect['h']

        # Image padding for particles that are close to the image borders:
        frame_padded = cv2.copyMakeBorder(temp_frame, h // 2, h // 2 + 1, w // 2, w // 2 + 1, cv2.BORDER_REFLECT)

        # Initialize:
        weights_new = np.empty(self.num_particles)
        particles_new = np.empty([self.num_particles, 2])
        
        # Get predicted particles and weights based on the sensor model:
        for i in range(self.num_particles):
            # Apply dynamics to predict the current time particle state:
            x_noisy = self.particles[i, 0] + np.random.normal(0, self.sigma_dyn)
            y_noisy = self.particles[i, 1] + np.random.normal(0, self.sigma_dyn)

            x_noisy = min(max(0, x_noisy), frame.shape[1] - 1)
            y_noisy = min(max(0, y_noisy), frame.shape[0] - 1)
            
            # Get noisy particles based on sensor model:
            particles_new[i, :] = [int(x_noisy), int(y_noisy)]
            
            # Get similarity (weights) based on the sensor model:
            frame_cutout = frame_padded[int(y_noisy) : int(y_noisy) + h, int(x_noisy) : int(x_noisy) + w]
            weights_new[i] = self.get_error_metric(self.template, frame_cutout)

        # Update particles for resampling
        self.particles = particles_new
        # Update normalized weights for resampling
        self.weights = weights_new / np.sum(weights_new)

        # USE IN PART3: class AppearanceModelPF(ParticleFilter):
        # Get the strongest particle with the highest weight 
        self.best_particle = self.particles[np.argmax(self.weights), :]

        # Resampling
        self.particles = self.resample_particles()


    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """

        x_weighted_mean = 0
        y_weighted_mean = 0

        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]

        # Complete the rest of the code as instructed.
        
        w = self.template_rect['w']
        h = self.template_rect['h']
        # Draw the rectangle of the tracking window:
        cv2.rectangle(frame_in, (int(x_weighted_mean) - w // 2, int(y_weighted_mean) - h // 2),
                      (int(x_weighted_mean) - w // 2 + w, int(y_weighted_mean) - h // 2 + h),
                      (0, 255, 255), 2)

        # Calculate the radius: sum_i {sqrt[(x_i - x_mean)**2 + (y_i - y_mean)**2] * weights_i} 
        x_y_mean = np.array([x_weighted_mean, y_weighted_mean], dtype=np.float64)
        distance_x_y = np.square(np.float64(self.particles) - x_y_mean)     
        distance_xy = np.sum(distance_x_y, axis=1)
        distance_weighted = np.multiply(np.sqrt(distance_xy), self.weights)
        circle_radius = np.sum(distance_weighted)

        # Draw the circle:
        cv2.circle(frame_in, (int(x_weighted_mean), int(y_weighted_mean)), int(circle_radius), (0, 255, 255), 2)



class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter.

        The documentation for this class is the same as the ParticleFilter
        above. There is one element that is added called alpha which is
        explained in the problem set documentation. By calling super(...) all
        the elements used in ParticleFilter will be inherited so you do not
        have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.alpha = kwargs.get('alpha')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)
        self.best_template = self.template
        
    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "Appearance Model" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """
        super(AppearanceModelPF, self).process(frame)  # call based class method

        w = self.template_rect['w']
        h = self.template_rect['h']

        # Image padding for particles that are close to the image borders:
        frame_padded = cv2.copyMakeBorder(frame, h // 2, h // 2 + 1, w // 2, w // 2 + 1, cv2.BORDER_REFLECT)

        y_best = int(self.best_particle[1])
        x_best = int(self.best_particle[0])
        self.best_template = frame_padded[y_best : y_best + h, x_best:x_best + w]

        self.template = self.alpha * self.best_template + (1 - self.alpha) * self.template
        self.template = np.uint8(self.template)

class MDParticleFilter(object):
    """MD Particle Filter."""

    def __init__(self, frame, template, **kwargs):
        # Parameters without default values:
        self.num_particles = kwargs.get('num_particles')  # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp')  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
        self.sigma_s = kwargs.get('sigma_s')
        self.template_rect = kwargs.get('template_coords')  # required by the autograder
        
        # Parameters with default values:
        self.alpha = kwargs.get('alpha', 0.0)

        # Other components:
        self.template = template
        self.frame = frame

        # Initialize your weights array. Read the docstring.
        self.weights = np.ones(self.num_particles) / self.num_particles  

        y_range, x_range = (frame.shape[0], frame.shape[1])
        particles_x = np.random.randint(x_range, size=self.num_particles)
        particles_y = np.random.randint(y_range, size=self.num_particles)
        particles_s = np.ones(self.num_particles)

        particles_x = particles_x[:, np.newaxis]
        particles_y = particles_y[:, np.newaxis]
        particles_s = particles_s[:, np.newaxis]
        self.particles = np.concatenate((particles_x, particles_y), axis=1)
        self.particles = np.concatenate((self.particles, particles_s), axis=1)  # Initialize your particles array. Read the docstring.

        w_temp = self.template_rect['w']
        h_temp = self.template_rect['h']
        x_temp = self.template_rect['x']
        y_temp = self.template_rect['y']
        # Intial position: the center of the template in the first frame.
        self.best_particle = [x_temp + w_temp // 2, y_temp + h_temp // 2, 1.]
        self.best_t = self.template

    def get_particles(self):
        """Returns the current particles state.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: particles data structure.
        """
        return self.particles


    def get_weights(self):
        """Returns the current particle filter's weights.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """
        return self.weights


    def get_error_metric(self, template, frame_cutout):
        """Returns the error metric used based on the similarity measure.

        Returns:
            float: similarity value.
        """
        template_temp = np.copy(template)
        frame_cutout_temp = np.copy(frame_cutout)

        if template.ndim == 3:
            template_temp = cv2.cvtColor(template_temp, cv2.COLOR_BGR2GRAY)
            frame_cutout_temp = cv2.cvtColor(frame_cutout_temp, cv2.COLOR_BGR2GRAY)

        m = template_temp.shape[0]
        n = template_temp.shape[1]

        array_temp = np.square(np.float64(template_temp) - np.float64(frame_cutout_temp))
        MSE = np.sum(array_temp) / (m * n)
        similarity_val = np.exp(- MSE / (2 * np.square(self.sigma_exp)))
        
        return similarity_val

    def resample_particles(self):
        """Returns a new set of particles

        This method does not alter self.particles.

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.

        Returns:
            numpy.array: particles data structure.
        """
        particles_index = np.arange(self.num_particles)
        particlesID = np.random.choice(particles_index, self.num_particles, p=self.weights)

        resampled_particles = np.empty([self.num_particles, 3])

        for i in range(self.num_particles):
            resampled_particles[i, :] = self.particles[particlesID[i], :]

        return resampled_particles


    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        temp_frame = np.copy(frame)

        w = self.template_rect['w']
        h = self.template_rect['h']

        pad_top = h // 2
        pad_bottom = h // 2 + 1
        pad_left = w // 2
        pad_right = w // 2 + 1

        # Image padding for particles that are close to the image borders:
        frame_padded = cv2.copyMakeBorder(temp_frame, h // 2, h // 2 + 1, w // 2, w // 2 + 1, cv2.BORDER_REFLECT)

        """Motion (dynamics) model"""
        # Initialize:
        weights_new = np.empty(self.num_particles)
        particles_new = np.empty([self.num_particles, 3])
        beta = 0.05
        # Get predicted particles and weights based on the sensor model:
        for i in np.arange(self.num_particles):
            s_noisy = self.particles[i, 2] - np.random.normal(0, self.sigma_s)  # float
            s_noisy = min(max(0.2, s_noisy), 1.0)
            x_noisy = self.particles[i, 0] + np.random.normal(0, s_noisy * self.sigma_dyn)     
            y_noisy = self.particles[i, 1] + np.random.normal(0, s_noisy * self.sigma_dyn)    
            
            if abs((x_noisy - np.float64(self.best_particle[0])) / np.float64(self.best_particle[0])) > 0.01:
                x_noisy = beta * self.particles[i, 0] + (1-beta) * self.best_particle[0] + np.random.normal(0, s_noisy * 0.1)

            if abs((y_noisy - np.float64(self.best_particle[1])) / np.float64(self.best_particle[1])) > 0.01:
                y_noisy = beta * self.particles[i, 1] + (1-beta) * self.best_particle[1] + np.random.normal(0, s_noisy * 0.1)


            particles_new[i, :] = [int(x_noisy), int(y_noisy), s_noisy]
            frame_cutout_resize = self.resize_template(x_noisy, y_noisy, s_noisy, frame_padded, False)
            weights_new[i] = self.get_error_metric(self.template, frame_cutout_resize)
            

        # Update particles for resampling
        self.particles = particles_new
        # Update normalized weights for resampling
        self.weights = weights_new / np.sum(weights_new)

        # Get the strongest particle with the highest weight 
        self.best_particle = self.particles[np.argmax(self.weights), :]

        # Resampling
        self.particles = self.resample_particles()
        
        best_t_templatesize = self.resize_template(self.best_particle[0], self.best_particle[1], self.best_particle[2], frame_padded, True)
        
        template_updated = self.alpha * best_t_templatesize + (1 - self.alpha) * self.template
        self.template = np.uint8(template_updated)

    def resize_template(self, x, y, s, frame_padded, update):
        w = self.template_rect['w']
        h = self.template_rect['h']

        pad_top = h // 2
        pad_left = w // 2
        
        frame_lb_x = int(x) + pad_left - int(round(pad_left * s))
        frame_ub_x = frame_lb_x + int(round(w * s))
        frame_lb_y = int(y) + pad_top - int(round(pad_top * s))
        frame_ub_y = frame_lb_y + int(round(h * s))
        frame_cutout = frame_padded[frame_lb_y : frame_ub_y, frame_lb_x : frame_ub_x]
        if update:
            self.best_t = frame_cutout 
        frame_cutout_resize = cv2.resize(np.copy(frame_cutout),(w, h))        
        return frame_cutout_resize
    
    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """
        w = self.template_rect['w']
        h = self.template_rect['h']

        x_weighted_mean = 0
        y_weighted_mean = 0
        w_weighted_mean = 0
        h_weighted_mean = 0

        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]
            w_weighted_mean += w * self.weights[i]
            h_weighted_mean += h * self.weights[i]
            cv2.circle(frame_in, (int(self.particles[i, 0]), int(self.particles[i, 1])), 4, (255, 0, 0), 2)

        cv2.rectangle(frame_in, (int(self.best_particle[0]) - int(self.best_t.shape[1]) // 2, int(self.best_particle[1]) - int(self.best_t.shape[0]) // 2),
                                (int(self.best_particle[0]) - int(self.best_t.shape[1]) // 2 + int(self.best_t.shape[1]),
                                 int(self.best_particle[1]) - int(self.best_t.shape[0]) // 2 + int(self.best_t.shape[0])),
                                (0, 255, 255), 2)
 
        # Calculate the radius: sum_i {sqrt[(x_i - x_mean)**2 + (y_i - y_mean)**2] * weights_i} 
        x_y_mean = np.array([x_weighted_mean, y_weighted_mean], dtype=np.float64)
        distance_x_y = np.square(np.float64(self.particles[:,:2]) - x_y_mean)     
        distance_xy = np.sum(distance_x_y, axis=1)
        distance_weighted = np.multiply(np.sqrt(distance_xy), self.weights)
        circle_radius = np.sum(distance_weighted)

        # Draw the circle:
        cv2.circle(frame_in, (int(x_weighted_mean), int(y_weighted_mean)), int(circle_radius), (0, 255, 255), 2)
