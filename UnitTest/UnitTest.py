import unittest
import numpy as np
import AlPyVision.Vision as alpyvision


class VisionUnitTest(unittest.TestCase):

    # Test Case 1: Find click positions
    def test_findClickPositions(self):

        # Unit test message
        message = 'Output 1 is not of instance list'

        # Instantiate new object
        vision = alpyvision.Vision()

        # Required parameters are: needle_img_path, haystack_img_path
        needle_img_path = 'resources/ores/test_ores_needle.png'
        haystack_img_path = 'resources/ores/test_ores_haystack.png'
        threshold = 0.85

        # Grab output & make comparison
        clickpoints = vision.findClickPositions(haystack_img_path, needle_img_path, threshold=threshold)
        self.assertIsInstance(obj=clickpoints, cls=list, msg=message)

    # Test Case 2: Get window information
    def test_getWindowInfo(self):
        # Unit test message
        message = 'Output 2 is not of instance list'
        message_2 = 'Output 2 is None at position 0'
        message_3 = 'Output 2 is None at position 1'
        message_4 = 'Output 2 is None at position 2'

        # Instantiate new object
        vision = alpyvision.Vision()

        # Grab output & run test
        window_info = vision.getWindowInfo()
        self.assertIsInstance(obj=window_info, cls=list, msg=message)
        self.assertIsNotNone(obj=window_info[0], msg=message_2)
        self.assertIsNotNone(obj=window_info[1], msg=message_3)
        self.assertIsNotNone(obj=window_info[2], msg=message_4)


    # Test Case 3: Capture Window
    def test_captureWindow(self):
        # Unit test message
        message = 'Output 3 is not of instance numpy.ndarray'

        # Instantiate new object & grab output
        vision = alpyvision.Vision()
        window_info = vision.getWindowInfo()
        window_capture = vision.captureWindow(window_info[0], window_info[1], window_info[2])

        # Run tests
        self.assertIsInstance(obj=window_capture, cls=np.ndarray, msg=message)


    # Test Case 4: Show bot vision
    def test_showBotVision(self):
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
