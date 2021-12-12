import unittest
import AlPyVision.Vision as alpyvision


class VisionUnitTest(unittest.TestCase):

    # Test Case 1: Find click positions
    def test_findClickPositions(self):

        # Instanciate new object
        vision = alpyvision.Vision()

        # Required parameters are: needle_img_path, haystack_img_path
        needle_img_path = 'resources/ores/test_ores_needle.png'
        haystack_img_path = 'resources/ores/test_ores_haystack.png'
        threshold = 0.85

        # Unit test message
        message = 'Output is not of instance list'

        # Grab output & make comparison
        clickpoints = vision.findClickPositions(haystack_img_path, needle_img_path, threshold=threshold, debug_mode='rectangles')
        self.assertIsInstance(obj=clickpoints, cls=list, msg=message)


if __name__ == '__main__':
    unittest.main()
