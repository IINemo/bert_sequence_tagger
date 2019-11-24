#!/usr/bin/env python

import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    return io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ).read()


setup(
    name='bert_sequence_tagger',
    version='0.1.0',
    description='A wrapper for hugging face transformers library',
    author='IINemo',
    author_email='',
    packages=['bert_sequence_tagger'],
    include_package_data=True,
    zip_safe=False,
    package_dir={'' : 'src'},
    install_requires=[
        'pytorch_transformers'
    ]
)
