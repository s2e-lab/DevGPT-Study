from setuptools import setup, find_packages
import os

VERSION = "0.4.1"


def get_long_description():
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md"),
        encoding="utf8",
    ) as fp:
        return fp.read()


setup(
    name="llm",
    description="Access large language models from the command-line",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Simon Willison",
    url="https://github.com/simonw/llm",
    project_urls={
        "Documentation": "https://llm.datasette.io/",
        "Issues": "https://github.com/simonw/llm/issues",
        "CI": "https://github.com/simonw/llm/actions",
        "Changelog": "https://github.com/simonw/llm/releases",
    },
    license="Apache License, Version 2.0",
    version=VERSION,
    packages=find_packages(),
    entry_points="""
        [console_scripts]
        llm=llm.cli:cli
    """,
    install_requires=[
        "click",
        "openai",
        "click-default-group-wheel",
        "sqlite-utils",
        "pydantic>=2.0.0",
        "PyYAML",
        "pluggy",
    ],
    extras_require={
        "test": [
            "pytest",
            "requests-mock",
            "cogapp",
            "mypy",
            "black",
            "ruff",
            "types-click",
            "types-PyYAML",
            "types-requests",
        ]
    },
    python_requires=">=3.7",
)
