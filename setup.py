import setuptools
from dataset_server.version import __version__



setuptools.setup(
    name="dataset_server",
    version=__version__,
    author="Dat Tran",
    author_email="viebboy@gmail.com",
    description="Toolkit to serve dataset in separate processes",
    long_description="Toolkit to serve dataset in separate processes",
    long_description_content_type="text",
    license='LICENSE.txt',
    packages=setuptools.find_packages(),
    classifiers=['Operating System :: POSIX', ],
)
