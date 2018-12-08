from setuptools import setup, find_packages

setup(name='transformer',

      version='0.1',

      url='https://github.com/GlassyWing/transformer-keras',

      license='Apache License 2.0',

      author='Manlier',

      author_email='dengjiaxim@gmail.com',

      description='inset pest predict model',

      packages=find_packages(exclude=['tests', 'examples']),

      package_data={'transformer': ['*.*', 'checkpoints/*', 'config/*']},

      long_description=open('README.md', encoding="utf-8").read(),

      zip_safe=False,

      install_requires=['pandas', 'matplotlib', 'keras'],

      )
