from setuptools import setup, find_packages
import os

package_name = 'image_processor_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Explicitly include only the .pth model file in the models directory
        (os.path.join('share', package_name, 'models'), [
            'image_processor_pkg/models/ImageReducer_bounded_grayscale_run_2.pth'
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='ROS 2 package for image processing with a deep learning model',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_processor = image_processor_pkg.image_processor:main',
        ],
    },
)

