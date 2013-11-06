#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from distutils.core import setup, Command
from setuptools import find_packages
from weka import classifiers as weka

class TestCommand(Command):
    description = "Runs unittests."
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        os.system('cd weka; python ./classifiers.py')

setup(name='weka',
    version=weka.__version__,
    description='A Python wrapper for the Weka data mining library.',
    author='Chris Spencer',
    author_email='chrisspen@gmail.com',
    url='https://github.com/chrisspen/weka',
    license='LGPL License',
    packages=find_packages(),
    classifiers = [
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: General",
    ],
    platforms=['OS Independent'],
    cmdclass={
        'test': TestCommand,
    },
)