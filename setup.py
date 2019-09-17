from setuptools import setup

setup(name='scikit-opt',
      version='0.3',
      description='Heuristic Algorithms in Python',
      url='https://github.com/guofei9987/scikit-opt',
      author='Guofei',
      author_email='guofei9987@foxmail.com',
      license='MIT',
      packages=['sko'],
      install_requires=['numpy', 'scipy', 'matplotlib', 'pandas'],
      zip_safe=False)
