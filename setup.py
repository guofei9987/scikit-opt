from setuptools import setup
from os import path as os_path

this_directory = os_path.abspath(os_path.dirname(__file__))


# 读取文件内容
def read_file(filename):
    with open(os_path.join(this_directory, filename), encoding='utf-8') as f:
        long_description = f.read()
    return long_description


# 获取依赖
def read_requirements(filename):
    return [line.strip() for line in read_file(filename).splitlines()
            if not line.startswith('#')]


setup(name='scikit-opt',
      python_requires='>=3.4.0',
      version='0.3.1',
      description='Heuristic Algorithms in Python',
      long_description=read_file('README.md'),
      long_description_content_type="text/markdown",
      url='https://github.com/guofei9987/scikit-opt',
      author='Guofei',
      author_email='guofei9987@foxmail.com',
      license='MIT',
      packages=['sko'],
      install_requires=['numpy', 'scipy', 'matplotlib', 'pandas'],
      zip_safe=False)
