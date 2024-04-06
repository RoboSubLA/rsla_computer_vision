from setuptools import find_packages, setup

package_name = 'ros2_cv'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('lib/' + package_name + '/models/',['models/experimental.py', 'models/common.py',
                                             'models/yolo.py']),
        ('lib/' + package_name + '/utils/',['utils/general.py', 'utils/torch_utils.py',
                                             'utils/plots.py', 'utils/datasets.py',
                                             'utils/google_utils.py', 'utils/activations.py',
                                             'utils/add_nms.py', 'utils/autoanchor.py',
                                             'utils/loss.py', 'utils/metrics.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='justin',
    maintainer_email='94007732+Scels12@users.noreply.github.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cv = cv.cv:main'
        ],
    },
)
