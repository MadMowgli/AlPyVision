import datetime
import os
import win32api
import pyautogui


class Data:
    # Fields
    resources = []
    directories = []
    current_directories = []
    left_mouse_button = 0
    right_mouse_button = 0

    screenshot_size_ores = None
    screenshot_correction_ores = None

    screenshot_size_trees = None
    screenshot_correction_trees = None

    screenshot_size_stones = None
    screenshot_correction_stones = None

    # Constructor
    def __init__(self):
        self.directories = ['/resources', '/resources/ores', '/resources/trees', '/resources/stones']
        self.resources = ['ores', 'trees', 'stones']
        self.left_mouse_button = 0x01
        self.right_mouse_button = 0x02
        self.screenshot_size_ores = (100, 100)
        self.screenshot_correction_ores = (60, 20)
        self.screenshot_size_trees = (80, 170)
        self.screenshot_correction_trees = (50, 120)
        self.screenshot_size_stones = (90, 80)
        self.screenshot_correction_stones = (50, 50)

    # Methods
    def gatherScreenshots(self, root_filepath):

        # Call to create the base folder structure if it doesn't exist yet
        self.createBaseFolderStructure(root_filepath)

        # Set counter for filenames
        counter = 0

        # Output instructions
        print('Thank you for using AlPyVision!')
        print('To gather screenshots we can use to train our machine learning model, please do the following: ')
        print('While clicking an ore you want to gather, hold down the Y key at the moment you click on the resource '
              'node.')
        print('While clicking a tree you want to gather, hold down the X key at the moment you click on the resource '
              'node.')
        print('While clicking a stone you want to gather, hold down the C key at the moment you click on the resource '
              'node.')
        print('Summarized: Y -> Ore, X -> Tree, C -> Stone')
        print('Finally, press P to exit the loop')

        # Capture ores
        run = True
        while run:

            # Check if left mouse button was clicked
            current_state_left = win32api.GetKeyState(self.left_mouse_button)
            current_state_Y = win32api.GetKeyState(ord('Y'))  # Key for ores
            current_state_X = win32api.GetKeyState(ord('X'))  # Key for trees
            current_state_C = win32api.GetKeyState(ord('C'))  # Key for stones

            # Capture ores
            # Check if left mouse button was clicked while holding down the Y key
            if current_state_left < 0 and current_state_Y < 0:
                # Get mouse position & screenshot region
                mouse_position = pyautogui.position()
                resource = self.resources[0]
                screenshot_region = self.getScreenshotRegion(mouse_position, resource)

                # Create screenshot with size for ores
                imageFileName = str(self.current_directories[0]) + '/captured_ore_{}.png'.format(counter)
                pyautogui.screenshot(imageFilename=imageFileName, region=screenshot_region)
                counter += 1
                print('Screenshot saved at {}!'.format(imageFileName))
                pyautogui.sleep(1)

            # Capture ores
            # Check if left mouse button was clicked while holding down the Y key
            if current_state_left < 0 and current_state_X < 0:
                resource = self.resources[1]
                # Get mouse position & screenshot region
                mouse_position = pyautogui.position()
                screenshot_region = self.getScreenshotRegion(mouse_position, resource)

                # Create screenshot with size for ores
                imageFileName = str(self.current_directories[1]) + '/captured_tree_{}.png'.format(counter)
                pyautogui.screenshot(imageFilename=imageFileName, region=screenshot_region)
                counter += 1
                print('Screenshot saved at {}!'.format(imageFileName))
                pyautogui.sleep(1)

            # Capture stones
            if current_state_left < 0 and current_state_C < 0:
                resource = self.resources[2]
                # Get mouse position & screenshot region
                mouse_position = pyautogui.position()
                screenshot_region = self.getScreenshotRegion(mouse_position, resource)

                # Create screenshot with size for ores
                imageFileName = str(self.current_directories[2]) + '/captured_stone_{}.png'.format(counter)
                pyautogui.screenshot(imageFilename=imageFileName, region=screenshot_region)
                counter += 1
                print('Screenshot saved at {}!'.format(imageFileName))
                pyautogui.sleep(1)

            # Check if y was pressed to escape the loop
            if win32api.GetAsyncKeyState(ord('P')):
                # Delete all empty directories
                self.cleanUp(root_filepath)
                run = False

    def getScreenshotRegion(self, mouse_position, resource):

        # Screenshot region for ores, assuming the player clicked around the middle of the ores
        if resource == self.resources[0]:
            top_left_x = mouse_position[0] - self.screenshot_correction_ores[0]
            top_left_y = mouse_position[1] - self.screenshot_correction_ores[1]
            width = self.screenshot_size_ores[0]
            height = self.screenshot_size_ores[1]
            return_tuple = (top_left_x, top_left_y, width, height)
            return return_tuple

        # Screenshot region for trees, assuming the player clicked on the stump of the tree
        if resource == self.resources[1]:
            top_left_x = mouse_position[0] - self.screenshot_correction_trees[0]
            top_left_y = mouse_position[1] - self.screenshot_correction_trees[1]
            width = self.screenshot_size_trees[0]
            height = self.screenshot_size_trees[1]
            return_tuple = (top_left_x, top_left_y, width, height)
            return return_tuple

        # Screenshot region for stones, assuming the player clicked around the middle of the stone
        if resource == self.resources[2]:
            top_left_x = mouse_position[0] - self.screenshot_correction_stones[0]
            top_left_y = mouse_position[1] - self.screenshot_correction_stones[1]
            width = self.screenshot_size_stones[0]
            height = self.screenshot_size_stones[1]
            return_tuple = (top_left_x, top_left_y, width, height)
            return return_tuple

    def createBaseFolderStructure(self, root_filepath):

        # Make root directory
        try:
            os.mkdir(root_filepath)
            print('Made fresh directory at ' + str(root_filepath))
        except FileExistsError:
            print('Directory already exists at ' + root_filepath)

        # Create folder for each resource
        for directory in self.directories:
            dirPath = str(root_filepath) + directory
            if not os.path.isdir(dirPath):
                os.mkdir(dirPath)

        # Create folders for current session
        subdirectories = self.directories[1:]
        today = datetime.date.today().strftime('%d-%m-%Y')
        counter = 0
        for subdir in subdirectories:
            dirPath = root_filepath + subdir + '/' + today
            while os.path.isdir(dirPath):
                # See if we already appended the _counter
                dirPath_year = dirPath[len(dirPath) - 4:]
                current_year = str(datetime.date.today().year)
                if current_year == dirPath_year:
                    dirPath = str(dirPath) + "_" + str(counter)
                else:
                    dirPath = dirPath[:-2] + "_" + str(counter)
                counter += 1
            os.mkdir(dirPath)
            self.current_directories.append(dirPath)
            counter = 0
            print('Fresh directory created at: ' + dirPath)

    def cleanUp(self, root_filepath):
        pass
        # Delete all empty directories
        # for directory in self.current_directories:
        #     scan = [x for x in os.scandir(directory)]
        #     if len(scan) == 0:
        #         os.rmdir(dirPath)
        #         print('Deleted empty directory at: ' + dirPath)
