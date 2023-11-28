from setuptools import setup
from setuptools import find_packages

setup(name='gym_carla',
      version='0.0.1',
      install_requires=['gym', 'pygame'],
      packages=find_packages()
)
