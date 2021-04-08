import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='torch_shapeguard',
    version='1.0.3',
    author="Rasmus Berg Palm",
    author_email="rasmusbergpalm@gmail.com",
    description="ShapeGuard allows you to very succinctly assert the expected shapes of tensors in a dynamic, einsum inspired way.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rasmusbergpalm/shapeguard",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
