import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

env_assets = package_files('robosumo/envs/assets')
policy_assets = package_files('robosumo/policy_zoo/assets')

setup(
    name='robosumo',
    version='0.0.1.dev',
    packages=find_packages(),
    description='RoboSumo MuJoCo environments with Gym API.',
    long_description=read('README.md'),
    url='https://github.com/openai/robosumo',
    install_requires=[
        'click', 'gym', 'mujoco_py>=1.5', 'numpy', 'tensorflow>=1.1.0',
    ],
    package_data={
        'robosumo': env_assets + policy_assets,
    },
)
