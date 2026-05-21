from setuptools import find_packages, setup


package_name = "bookarm_control_py"


setup(
    name=package_name,
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (
            f"share/{package_name}/launch",
            [
                "launch/grasp_book.launch.py",
                "launch/place_book_on_disk.launch.py",
            ],
        ),
    ],
    install_requires=[],
    scripts=[
        "ros2_scripts/grasp_book",
        "ros2_scripts/place_book_on_disk",
    ],
    zip_safe=False,
    entry_points={
        "console_scripts": [],
    },
)
