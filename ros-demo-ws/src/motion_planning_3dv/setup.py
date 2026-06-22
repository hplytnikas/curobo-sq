from setuptools import find_packages, setup
from glob import glob


package_name = 'motion_planning_3dv'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + "/launch", glob('launch/*launch.[pxy][yma]*')),
        ('share/' + package_name + "/config", glob('config/*')),
        ('share/' + package_name + "/config", glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vision',
    maintainer_email='valentinveluppillai@gmail.com',
    description='Collection of nodes for a small manipulation demonstration',
    license='TBD',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'curobo_ros_executor = motion_planning_3dv.curobo_ros_executor:main',
            'sq_renderer = motion_planning_3dv.sq_renderer:main',
            'sq_perception = motion_planning_3dv.sq_perception:main',
            'depth_image_m_to_mm = motion_planning_3dv.depth_image_m_to_mm:main',
            'scene_manager = motion_planning_3dv.scene_manager:main',
            'demo = motion_planning_3dv.demo:main',
        ],
    },
)
