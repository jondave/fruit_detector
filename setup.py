from setuptools import find_packages, setup

package_name = 'fruit_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cheddar',
    maintainer_email='jcox289@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
<<<<<<< HEAD
=======
            'fruit_detector = fruit_detector.fruit_detector:main'
>>>>>>> d1b82ce (Initial commit)
        ],
    },
)
