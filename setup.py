from setuptools import setup, find_packages

setup(name='transformer-keras',

      version='0.1-SNAPSHOT',

      url='https://github.com/GlassyWing/transformer-keras',

      license='Apache License 2.0',

      author='Manlier',

      author_email='dengjiaxim@gmail.com',

      description='transformer keras version',

      packages=find_packages(exclude=['tests', 'examples']),

      package_data={'transformer-keras': ['*.*', 'checkpoints/*', 'config/*']},

      long_description=open('README.md', encoding="utf-8").read(),

      zip_safe=False,

      install_requires=['pandas', 'matplotlib', 'keras'],

      )
