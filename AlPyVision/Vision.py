import numpy as np
import cv2 as cv

class Vision:

    # Attributes
    match_template_methods = None
    normalized_match_template_methods = None
    debug_modes = [None, 'rectangles', 'crosshairs']

    # Constructor
    def __init__(self):
        # The match template methods are stored as strings to improve performance
        self.match_template_methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED',
                                       'cv.TM_CCORR', 'cv.TM_CCORR_NORMED',
                                       'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
        self.normalized_match_template_methods = ['cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF_NORMED']

    # Methods
    def findClickPositions(self, needle_img_path, haystack_img_path, method='cv.CCOEFF_NORMED', threshold=0.5,
                           debug_mode=None, ):
        '''
        This method performs the openCV.matchTemplate()-method, using the needle_img on the haystack_img.
        Heavily inspired and mostly inherited from: https://github.com/learncodebygaming/opencv_tutorials/blob/master/003_group_rectangles/main.py
        :param needle_img_path:
        :param haystack_img_path:
        :param threshold:
        :param debug_mode:
        :return click_points:
        '''

        # Local constants
        group_threshold = 1
        group_eps = 0.5
        line_color = (0, 255, 0)
        line_type = cv.LINE_4
        line_thickness = 2
        marker_color = line_color
        marker_type = cv.MARKER_CROSS
        marker_size = 40

        # Local variables
        haystack_img = None
        needle_img = None
        needle_img_width = 0
        needle_img_height = 0
        rectangles = []
        click_points = []

        # Validate user inputs from parameters
        if method not in self.match_template_methods:
            method = self.match_template_methods[0]

        if debug_mode not in self.debug_modes:
            debug_mode = self.debug_modes[1]


        # Try to load images in memory using cv.imread()
        try:
            haystack_img = cv.imread(haystack_img_path)
            needle_img = cv.imread(needle_img_path)
        except BaseException as exception:
            print('[VISION] Exception while loading images using cv.imread()')
            print('[VISION] Exception info: ', exception)

        # Perform the matchTemplate function
        result_matrix = cv.matchTemplate(haystack_img, needle_img, method)

        # Grab positions from the match result matrix that exceed the given threshold
        locations = np.where(result_matrix >= threshold)

        # Transform the output from np.where to an array of tuples, containing our (X, Y) coordinates
        locations = list(zip(*locations[::-1]))

        # Create a list of rectangles out of the locations we found. Append each rectangle twice to escape the
        # cv.groupRectanges() function eliminating single rectangles, since we want them as well.
        for location in locations:
            rectangle = [int(location[0]), int(location[1]), needle_img_width, needle_img_height]
            rectangles.append(rectangle)
            rectangles.append(rectangle)

        # Group the rectangles so we get cleaner output
        rectangles, weights = cv.groupRectangles(rectangles, groupThreshold=group_threshold, eps=group_eps)

        # Get the center point from each rectangle
        for (x_coordinate, y_coordinate, width, heigth) in rectangles:
            center_x = x_coordinate + int(width / 2)
            center_y = y_coordinate + int(heigth / 2)
            coordinate_tuple = (center_x, center_y)
            click_points.append(coordinate_tuple)

            # Draw rectangles if we're in rectangles-debug mode
            if debug_mode == self.debug_modes[1]:
                top_left = (x_coordinate, y_coordinate)
                bottom_right = (x_coordinate + width, y_coordinate + heigth)
                cv.rectangle(haystack_img, top_left, bottom_right,
                             color=line_color, lineType=line_type,
                             thickness=line_thickness)

            # Draw crosshairs if we're in crosshairs-debug mode
            elif debug_mode == self.debug_modes[2]:
                cv.drawMarker(haystack_img, (center_x, center_y),
                              color=marker_color, marker_type=marker_type,
                              markerSize=marker_size, thickness=line_thickness)

        # Show outputs
        if debug_mode:
            cv.imshow('Matches', haystack_img)
            cv.waitKey()

        return click_points
