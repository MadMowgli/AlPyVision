from time import time
import numpy as np
import cv2 as cv
import win32gui, win32ui, win32con


class Vision:
    # Attributes
    match_template_methods = list()
    normalized_match_template_methods = list()
    debug_modes = list()
    copper_ore_rgb = list()
    tin_ore_rgb = list()

    # --------------- Constructor
    def __init__(self):
        self.match_template_methods = [cv.TM_CCOEFF, cv.TM_CCOEFF_NORMED,
                                       cv.TM_CCORR, cv.TM_CCORR_NORMED,
                                       cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]
        self.normalized_match_template_methods = [cv.TM_CCOEFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_SQDIFF_NORMED]
        self.debug_modes = [None, 'rectangles', 'crosshairs', 'visual']

        # Set colour codes
        self.copper_ore_rgb.extend([(255, 228, 154), (201, 163, 111), (149, 126, 86)])
        self.tin_ore_rgb.extend([(92, 253, 152), (33, 147, 38), (59, 196, 119)])

    # --------------- Methods

    # Method 1: Finding click positions based on the cv.matchTemplate() function
    def findClickPositions(self, haystack_img, needle_img, method=cv.TM_CCOEFF_NORMED, threshold=0.9,
                           debug_mode=None):
        '''
        This method performs the openCV.matchTemplate()-method, using the needle_img on the haystack_img.
        Heavily inspired and mostly inherited from: https://github.com/learncodebygaming/opencv_tutorials/blob/master/003_group_rectangles/main.py
        :param haystack_image:
        :param needle_image:
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
        needle_img_width = 0
        needle_img_height = 0
        rectangles = []
        click_points = []

        # Validate user inputs from parameters
        if method not in self.match_template_methods:
            method = self.match_template_methods[0]

        if debug_mode not in self.debug_modes:
            debug_mode = self.debug_modes[1]

        # We're already passing in image data from cv.imread(), get image shape now
        try:
            needle_img_width = needle_img.shape[0]
            needle_img_height = needle_img.shape[1]
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
        for (x_coordinate, y_coordinate, width, height) in rectangles:
            center_x = x_coordinate + int(width / 2)
            center_y = y_coordinate + int(height / 2)
            coordinate_tuple = (center_x, center_y)
            click_points.append(coordinate_tuple)

            # Draw rectangles if we're in rectangles-debug mode
            if debug_mode == self.debug_modes[1]:
                top_left = (x_coordinate, y_coordinate)
                bottom_right = (x_coordinate + width, y_coordinate + height)
                cv.rectangle(haystack_img, top_left, bottom_right,
                             color=line_color, lineType=line_type,
                             thickness=line_thickness)
                cv.waitKey()

            # Draw crosshairs if we're in crosshairs-debug mode
            elif debug_mode == self.debug_modes[2]:
                cv.drawMarker(haystack_img, (center_x, center_y),
                              color=marker_color, marker_type=marker_type,
                              markerSize=marker_size, thickness=line_thickness)
                cv.waitKey()

        # Show outputs
        if debug_mode == self.debug_modes[3]:
            return haystack_img

        return click_points

    # Method 2: Return window handle, window_width and window_height of a given window
    def getWindowInfo(self, window_name='Albion Online Client'):

        # Local variables
        window_information = []

        # Local variables
        window_handle = None
        window_width = None
        window_height = None

        # Try grabbing the albion online client window_handle and setting the width and height
        try:
            window_handle = win32gui.FindWindow(None, window_name)
            window_rectangle = win32gui.GetWindowRect(window_handle)
            window_width = window_rectangle[2] - window_rectangle[0]
            window_height = window_rectangle[3] - window_rectangle[1]
        except BaseException as exception:
            print('[VISION] Exception capturing a window_handle using win32gui.FindWindow()')
            print('[VISION] Exception info: ', exception)

        # Add results to the result array
        window_information.extend([window_handle, window_width, window_height])

        return window_information

    # Method 3: Capturing the window from Albion and returning it in an openCV-understandable format
    def captureWindow(self, window_width, window_height, window_handle=None, debug_mode=False):

        # Parse out window handle - if it is none, we capture the whole desktop
        # IMPORTANT: To capture the Albion client, we need to set window_handle to None, since we only capture the
        # initial screenshot and never update it. See: https://www.youtube.com/watch?v=7k4j-uL8WSQ&list=PL1m2M8LQlzfKtkKq2lK5xko4X-8EZzFPI&t=132s
        if window_handle is None:
            window_handle = win32gui.GetDesktopWindow()

        # Get window device context - a structure that defines a set of graphic objects and their associated attributes
        # https://docs.microsoft.com/en-us/windows/win32/gdi/device-contexts
        window_device_context = win32gui.GetWindowDC(window_handle)
        device_context = win32ui.CreateDCFromHandle(window_device_context)
        compatible_device_context = device_context.CreateCompatibleDC()

        # Creates a bitmap out of the device context and convert it into a format openCV can read
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(device_context, window_width, window_height)

        compatible_device_context.SelectObject(bitmap)
        compatible_device_context.BitBlt((0, 0), (window_width, window_height), device_context,
                                         (0, 0), win32con.SRCCOPY)

        bitmap_bits = bitmap.GetBitmapBits(True)
        img = np.frombuffer(bitmap_bits, dtype='uint8')
        img.shape = (window_height, window_width, 4)

        # Free resources
        device_context.DeleteDC()
        compatible_device_context.DeleteDC()
        win32gui.ReleaseDC(window_handle, window_device_context)
        win32gui.DeleteObject(bitmap.GetHandle())

        # Drop alpha channel to avoid cv.matchTemplate() error
        img = img[..., :3]

        # Make image C_CONTIGUOUS to avoid typeErrors
        img = np.ascontiguousarray(img)

        if debug_mode:
            cv.imshow('Bot Vision', img)
            cv.waitKey()

        return img

    # Method 4: Showing the bot-vision
    def showBotVision(self, image, show_fps=True):
        cv.imshow('Bot Vision', image)

        if cv.waitKey(1) == ord('y'):
            cv.destroyAllWindows()
            return False
