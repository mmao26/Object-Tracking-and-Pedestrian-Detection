"""Object Tracking and Pedestrian Detection"""
### Author Name: Manqing Mao
### GTID: mmao33
import cv2
import filters
import os
import numpy as np

# I/O directories
input_dir = "input_images"
output_dir = "output"

NOISE_1 = {'x': 2.5, 'y': 2.5}
NOISE_2 = {'x': 7.5, 'y': 7.5}


# Helper code
def run_particle_filter(filter_class, imgs_dir, template_rect,
                        save_frames={}, **kwargs):
    """Runs a particle filter on a given video and template.

    Create an object of type pf_class, passing in initial video frame,
    template (extracted from first frame using template_rect), and any
    keyword arguments.

    Do not modify this function except for the debugging flag.

    Args:
        filter_class (object): particle filter class to instantiate
                           (e.g. ParticleFilter).
        imgs_dir (str): path to input images.
        template_rect (dict): template bounds (x, y, w, h), as float
                              or int.
        save_frames (dict): frames to save
                            {<frame number>|'template': <filename>}.
        **kwargs: arbitrary keyword arguments passed on to particle
                  filter class.

    Returns:
        None.
    """

    imgs_list = [f for f in os.listdir(imgs_dir)
                 if f[0] != '.' and f.endswith('.jpg')]
    imgs_list.sort()

    # Initialize objects
    template = None
    pf = None
    frame_num = 0

    # Loop over video (till last frame or Ctrl+C is presssed)
    for img in imgs_list:

        frame = cv2.imread(os.path.join(imgs_dir, img))

        # Extract template and initialize (one-time only)
        if template is None:
            template = frame[int(template_rect['y']):
                             int(template_rect['y'] + template_rect['h']),
                             int(template_rect['x']):
                             int(template_rect['x'] + template_rect['w'])]

            if 'template' in save_frames:
                cv2.imwrite(save_frames['template'], template)

            pf = filter_class(frame, template, **kwargs)

        # Process frame
        pf.process(frame)

        if True:  # For debugging, it displays every frame
            out_frame = frame.copy()
            pf.render(out_frame)
            cv2.imshow('Tracking', out_frame)
            cv2.waitKey(1)

        # Render and save output, if indicated
        if frame_num in save_frames:
            frame_out = frame.copy()
            pf.render(frame_out)
            cv2.imwrite(save_frames[frame_num], frame_out)

        # Update frame number
        frame_num += 1
        if frame_num % 20 == 0:
            print 'Working on frame %d' % frame_num

def run_particle_filter_part5(filter_class, imgs_dir, start_frame, end_frame, template_rect,
                        save_frames={}, **kwargs):

    imgs_list = [f for f in os.listdir(imgs_dir)
                 if f[0] != '.' and f.endswith('.jpg')]
    imgs_list.sort()

    imgs_list = imgs_list[start_frame-1 : end_frame]

    # Initialize objects
    template = None
    pf = None
    frame_num = 0

    # Loop over video (till last frame or Ctrl+C is presssed)
    for img in imgs_list:

        frame = cv2.imread(os.path.join(imgs_dir, img))

        # Extract template and initialize (one-time only)
        if template is None:
            template = frame[int(template_rect['y']):
                             int(template_rect['y'] + template_rect['h']),
                             int(template_rect['x']):
                             int(template_rect['x'] + template_rect['w'])]

            if 'template' in save_frames:
                cv2.imwrite(save_frames['template'], template)

            pf = filter_class(frame, template, **kwargs)

        # Process frame
        pf.process(frame)

        if True:  # For debugging, it displays every frame
            out_frame = frame.copy()
            pf.render(out_frame)
            cv2.imshow('Tracking', out_frame)
            cv2.waitKey(1)

        # Render and save output, if indicated
        if frame_num in save_frames:
            frame_out = frame.copy()
            pf.render(frame_out)
            cv2.imwrite(save_frames[frame_num], frame_out)

        # Update frame number
        frame_num += 1
        if frame_num % 20 == 0:
            print 'Working on frame %d' % frame_num

def run_kalman_filter(kf, imgs_dir, noise, sensor, save_frames={},
                      template_loc=None):

    imgs_list = [f for f in os.listdir(imgs_dir)
                 if f[0] != '.' and f.endswith('.jpg')]
    imgs_list.sort()

    frame_num = 0

    if sensor == "hog":
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    elif sensor == "matching":
        frame = cv2.imread(os.path.join(imgs_dir, imgs_list[0]))
        template = frame[template_loc['y']:
                         template_loc['y'] + template_loc['h'],
                         template_loc['x']:
                         template_loc['x'] + template_loc['w']]

    else:
        raise ValueError("Unknown sensor name. Choose between 'hog' or "
                         "'matching'")

    for img in imgs_list:

        frame = cv2.imread(os.path.join(imgs_dir, img))

        # Sensor
        if sensor == "hog":
            (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
                                                    padding=(8, 8), scale=1.05)

            if len(weights) > 0:
                max_w_id = np.argmax(weights)
                z_x, z_y, z_w, z_h = rects[max_w_id]

                z_x += z_w // 2
                z_y += z_h // 2

                z_x += np.random.normal(0, noise['x'])
                z_y += np.random.normal(0, noise['y'])

        elif sensor == "matching":
            corr_map = cv2.matchTemplate(frame, template, cv2.cv.CV_TM_SQDIFF)
            z_y, z_x = np.unravel_index(np.argmin(corr_map), corr_map.shape)

            z_w = template_loc['w']
            z_h = template_loc['h']

            z_x += z_w // 2 + np.random.normal(0, noise['x'])
            z_y += z_h // 2 + np.random.normal(0, noise['y'])

        x, y = kf.process(z_x, z_y)

        if False:  # For debugging, it displays every frame
            out_frame = frame.copy()
            cv2.circle(out_frame, (int(z_x), int(z_y)), 20, (0, 0, 255), 2)
            cv2.circle(out_frame, (int(x), int(y)), 10, (255, 0, 0), 2)
            cv2.rectangle(out_frame, (int(z_x) - z_w // 2, int(z_y) - z_h // 2),
                          (int(z_x) + z_w // 2, int(z_y) + z_h // 2),
                          (0, 0, 255), 2)

            cv2.imshow('Tracking', out_frame)
            cv2.waitKey(1)

        # Render and save output, if indicated
        if frame_num in save_frames:
            frame_out = frame.copy()
            cv2.circle(frame_out, (int(x), int(y)), 10, (255, 0, 0), 2)
            cv2.imwrite(save_frames[frame_num], frame_out)

        # Update frame number
        frame_num += 1
        if frame_num % 20 == 0:
            print 'Working on frame %d' % frame_num



def part_1b():
    print "Part 1b"

    template_loc = {'y': 72, 'x': 140, 'w': 50, 'h': 50}

    # Define process and measurement arrays if you want to use other than the
    # default. Pass them to KalmanFilter.
    Q = None  # Process noise array
    R = None  # Measurement noise array

    kf = filters.KalmanFilter(template_loc['x'], template_loc['y'])

    save_frames = {10: os.path.join(output_dir, 'filters-1-b-1.png'),
                   30: os.path.join(output_dir, 'filters-1-b-2.png'),
                   59: os.path.join(output_dir, 'filters-1-b-3.png'),
                   99: os.path.join(output_dir, 'filters-1-b-4.png')}

    run_kalman_filter(kf,
                      os.path.join(input_dir, "circle"),
                      NOISE_2,
                      "matching",
                      save_frames,
                      template_loc)


def part_1c():
    print "Part 1c"

    init_pos = {'x': 311, 'y': 217}

    # Define process and measurement arrays if you want to use other than the
    # default. Pass them to KalmanFilter.
    Q = None  # Process noise array
    R = None  # Measurement noise array

    kf = filters.KalmanFilter(init_pos['x'], init_pos['y'])

    save_frames = {10: os.path.join(output_dir, 'filters-1-c-1.png'),
                   33: os.path.join(output_dir, 'filters-1-c-2.png'),
                   84: os.path.join(output_dir, 'filters-1-c-3.png'),
                   159: os.path.join(output_dir, 'filters-1-c-4.png')}

    run_kalman_filter(kf,
                      os.path.join(input_dir, "walking"),
                      NOISE_1,
                      "hog",
                      save_frames)


def part_2a():

    template_loc = {'y': 72, 'x': 140, 'w': 50, 'h': 50}

    save_frames = {10: os.path.join(output_dir, 'filters-2-a-1.png'),
                   30: os.path.join(output_dir, 'filters-2-a-2.png'),
                   59: os.path.join(output_dir, 'filters-2-a-3.png'),
                   99: os.path.join(output_dir, 'filters-2-a-4.png')}

    num_particles = 100  # Define the number of particles
    sigma_mse = 10  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 10  # Define the value of sigma for the particles movement (dynamics)

    run_particle_filter(filters.ParticleFilter,  # particle filter model class
                        os.path.join(input_dir, "circle"),
                        template_loc,
                        save_frames,
                        num_particles=num_particles, sigma_exp=sigma_mse,
                        sigma_dyn=sigma_dyn,
                        template_coords=template_loc)  # Add more if you need to


def part_2b():

    template_loc = {'x': 360, 'y': 141, 'w': 127, 'h': 179}

    save_frames = {10: os.path.join(output_dir, 'filters-2-b-1.png'),
                   33: os.path.join(output_dir, 'filters-2-b-2.png'),
                   84: os.path.join(output_dir, 'filters-2-b-3.png'),
                   99: os.path.join(output_dir, 'filters-2-b-4.png')}

    num_particles = 1000  # Define the number of particles
    sigma_mse = 10  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 20  # Define the value of sigma for the particles movement (dynamics)

    run_particle_filter(filters.ParticleFilter,  # particle filter model class
                        os.path.join(input_dir, "pres_debate_noisy"),
                        template_loc,
                        save_frames,
                        num_particles=num_particles, sigma_exp=sigma_mse,
                        sigma_dyn=sigma_dyn,
                        template_coords=template_loc)  # Add more if you need to


def part_3():
    template_rect = {'x': 538, 'y': 377, 'w': 73, 'h': 117}

    save_frames = {22: os.path.join(output_dir, 'filters-3-a-1.png'),
                   50: os.path.join(output_dir, 'filters-3-a-2.png'),
                   160: os.path.join(output_dir, 'filters-3-a-3.png')}

    num_particles = 1000  # Define the number of particles
    sigma_mse = 10  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 10  # Define the value of sigma for the particles movement (dynamics)
    alpha = 0.2  # Set a value for alpha

    run_particle_filter(filters.AppearanceModelPF,  # particle filter model class
                        os.path.join(input_dir, "pres_debate"),
                        # input video
                        template_rect,
                        save_frames,
                        num_particles=num_particles, sigma_exp=sigma_mse,
                        sigma_dyn=sigma_dyn, alpha=alpha,
                        template_coords=template_rect)  # Add more if you need to


def part_4():
    template_rect = {'x': 210, 'y': 37, 'w': 103, 'h': 285}

    save_frames = {40: os.path.join(output_dir, 'filters-4-a-1.png'),
                   100: os.path.join(output_dir, 'filters-4-a-2.png'),
                   240: os.path.join(output_dir, 'filters-4-a-3.png'),
                   300: os.path.join(output_dir, 'filters-4-a-4.png')}

    num_particles = 400  # Define the number of particles
    sigma_md = 10  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 20  # Define the value of sigma for the particles movement (dynamics)
    sigma_s = 0.02
    
    run_particle_filter(filters.MDParticleFilter,
                        os.path.join(input_dir, "pedestrians"),
                        template_rect,
                        save_frames,
                        num_particles=num_particles, sigma_exp=sigma_md,
                        sigma_dyn=sigma_dyn, sigma_s=sigma_s,
                        template_coords=template_rect)  # Add more if you need to


def part_5():
    """Tracking multiple Targets.

    Use either a Kalman or particle filter to track multiple targets
    as they move through the given video.  Use the sequence of images
    in the TUD-Campus directory.

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    
    template_loc1 = {'y': 215, 'x': 120, 'w': 100, 'h': 200}

    save_frames1 = {28: os.path.join(output_dir, 'filters-5-f28-1.png'),
                   55: os.path.join(output_dir, 'filters-5-f55-1.png')}

    num_particles1 = 500  # Define the number of particles
    sigma_mse1 = 10  # Define the value of sigma for the measurement exponential equation
    sigma_dyn1 = 10  # Define the value of sigma for the particles movement (dynamics)

    run_particle_filter_part5(filters.ParticleFilter,  # particle filter model class
                        os.path.join(input_dir, "TUD-Campus"),
                        1,
                        65,
                        template_loc1,
                        save_frames1,
                        num_particles=num_particles1, sigma_exp=sigma_mse1,
                        sigma_dyn=sigma_dyn1,
                        template_coords=template_loc1)  # Add more if you need to
    
    
    template_loc2 = {'y': 200, 'x': 340, 'w': 100, 'h': 200}
    
    save_frames2 = {28: os.path.join(output_dir, 'filters-5-f28-2.png')}

    num_particles2 = 1000  # Define the number of particles
    sigma_mse2 = 15  # Define the value of sigma for the measurement exponential equation
    sigma_dyn2 = 15  # Define the value of sigma for the particles movement (dynamics)
    
    run_particle_filter_part5(filters.ParticleFilter,  # particle filter model class
                        os.path.join(input_dir, "TUD-Campus"),
                        1,
                        50,
                        template_loc2,
                        save_frames2,
                        num_particles=num_particles2, sigma_exp=sigma_mse2,
                        sigma_dyn=sigma_dyn2,
                        template_coords=template_loc2)  # Add more if you need to
    
    template_loc3 = {'y': 200, 'x': 20, 'w': 100, 'h': 200}
    
    save_frames3 = {5: os.path.join(output_dir, 'filters-5-f28-3.png'),
                   32: os.path.join(output_dir, 'filters-5-f55-3.png'),
                   46: os.path.join(output_dir, 'filters-5-f70-3.png')}

    
    num_particles3 = 500  # Define the number of particles
    sigma_mse3 = 8.5  # Define the value of sigma for the measurement exponential equation
    sigma_dyn3 = 8.5  # Define the value of sigma for the particles movement (dynamics)
    
    run_particle_filter_part5(filters.ParticleFilter,  # particle filter model class
                        os.path.join(input_dir, "TUD-Campus"),
                        24,
                        70,
                        template_loc3,
                        save_frames3,
                        num_particles=num_particles3, sigma_exp=sigma_mse3,
                        sigma_dyn=sigma_dyn3,
                        template_coords=template_loc3)  # Add more if you need to
    
    frame28_1 = cv2.imread(os.path.join(output_dir, 'filters-5-f28-1.png'))
    frame28_2 = cv2.imread(os.path.join(output_dir, 'filters-5-f28-2.png'))
    frame28_3 = cv2.imread(os.path.join(output_dir, 'filters-5-f28-3.png'))
    image28_temp = cv2.addWeighted(frame28_1, 0.5, frame28_2,0.5, 0)
    image28 = cv2.addWeighted(image28_temp, 0.5, frame28_3,0.5, 0)
    cv2.imwrite("filters-5-a-1.png", image28)

    frame55_1 = cv2.imread(os.path.join(output_dir, 'filters-5-f55-1.png'))
    frame55_3 = cv2.imread(os.path.join(output_dir, 'filters-5-f55-3.png'))
    image55 = cv2.addWeighted(frame55_1, 0.5, frame55_3, 0.5, 0)
    cv2.imwrite("filters-5-a-2.png", image55)
    
    image70 = cv2.imread(os.path.join(output_dir, 'filters-5-f70-3.png'))
    cv2.imwrite("filters-5-a-3.png", image70)
    
def part_6():
    """Tracking pedestrians from a moving camera.

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    raise NotImplementedError

if __name__ == '__main__':
    part_1b()
    
    part_1c()

    part_2a()
    part_2b()
    part_3()
    part_4()
    part_5()

    #part_6()
    

