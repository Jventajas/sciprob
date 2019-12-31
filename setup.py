from setuptools import setup, find_packages

setup(name='sciprob',
      version='0.01',
      url='https://github.com/Jventajas/sciprob',
      license='MIT',
      author='Javi Ventajas',
      author_email='jvhernandez445@gmail.com',
      description='The pedagogical ML library.',
      packages=find_packages(exclude=['tests']),
      long_description=open('README.md').read(),
      zip_safe=False)
