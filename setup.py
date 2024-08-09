from setuptools import setup, find_packages

with open('README.md', 'r', encoding="utf-8") as f:
    readme = f.read()

version = '0.0.0'
setup(
    name='pse',
    packages=find_packages(exclude=["hf_space", "tests"]),
    version=version,
    license='MIT',
    description='PSE',
    url='https://github.com/asahi417',
    keywords=['machine-learning', 'computational-art'],
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Asahi Ushio',
    author_email='asahi1992ushio@gmail.com',
    classifiers=[
        'Development Status :: 4 - Beta',       # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',      # Define that your audience are developers
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',   # Again, pick a license
        'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    ],
    include_package_data=True,
    test_suite='tests',
    install_requires=[
        "datasets",
        "transformers",
        "accelerate",
        "safetensors",
        "sentencepiece",
        "protobuf",
        "bm25s",
        "sentence_transformers"
        "torch"
    ],
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [],
    }
)