from setuptools import find_packages, setup

package_name = 'snakebot_py_code'

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
    maintainer='hahnlam',
    maintainer_email='hahnlon.lam@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "simple_publisher = snakebot_py_code.simple_publisher:main",
            "simple_subscriber = snakebot_py_code.simple_subscriber:main",
            "parameter = snakebot_py_code.parameter:main"
        ],
    },
)
