import setuptools

# Set the contents of the readme file as long_description
with open('README.md', 'r') as readme:
    long_description = readme.read()

# Let setuptools work
setuptools.setup(
    name='alpyvision',
    version='0.0.1',
    author='Joel Läubin',
    author_email='laeubin.j@protonmail.com',
    description='AlPyVision is a Python wrapper to work with openCV and the Albion Online game client.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['albion', 'opencv', 'computer vision'],
    url='https://github.com/MadMowgli/AlPyVision',
    packages=setuptools.find_packages(),
    install_requires=['opencv-python', 'numpy'],
    license='MIT',
    classifiers=[
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Operating System :: Microsoft :: Windows'
    ]
)

