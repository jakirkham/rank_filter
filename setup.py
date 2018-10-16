import os
import sys
from glob import glob

import setuptools
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from distutils.sysconfig import get_config_var, get_python_inc
from distutils.version import LooseVersion

import versioneer


assert LooseVersion(setuptools.__version__) >= LooseVersion("18.0"), \
        "Requires `setuptools` version 18.0 or higher."

class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

def readme():
    with open("README.rst", "r") as f:
        return(f.read())


version = versioneer.get_version()

with open("src/version.pxi", "w") as f:
     f.writelines([
         "__version__ = " + "\"" + str(version) + "\""
     ])

cython_dep = ["cython >= 0.23"]
numpy_dep = ["numpy >= 1.7"]
boost_dep = ["boost-cpp >= 1.56"]
boost_dep = (boost_dep if sys.argv[1] == "bdist_conda" else [])

setup_requires = cython_dep + numpy_dep
setup_requires = setup_requires if (sys.argv[1].startswith("bdist") or
                                    sys.argv[1].startswith("build") or
                                    sys.argv[1].startswith("install")) else []
build_requires = cython_dep + numpy_dep + boost_dep
install_requires = numpy_dep + boost_dep
install_requires += [] if sys.argv[1] == "bdist_conda" else cython_dep
tests_require = cython_dep + numpy_dep

include_dirs = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "include"),
    os.path.dirname(get_python_inc()),
    get_python_inc()
]
library_dirs = list(filter(
    lambda v: v is not None,
    [get_config_var("LIBDIR")]
))
sources = glob("src/*.pxd") + glob("src/*.pyx")

libraries = []
if os.name == "posix":
    libraries.append("boost_container")
elif os.name == "nt":
    libname = "boost_container"

    path = os.environ.get("LIB", "").split(";")

    libmatches = sum(
        list(glob(os.path.join(p, "%s*.lib" % libname)) for p in path), []
    )
    library_dirs.append(os.path.dirname(libmatches[0]))
    libraries.append(os.path.splitext(os.path.basename(libmatches[0]))[0])

define_macros = [
    ("CYTHON_TRACE", "1"),
    ("CYTHON_TRACE_NOGIL", "1"),
]

extra_compile_args = []


setup(
    name="rank_filter",
    version=version,
    description="A simple python module containing an in-place linear rank"
                " filter optimized in C++.",
    long_description=readme(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: C++',
        'Programming Language :: Cython',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries'
    ],
    author="John Kirkham",
    author_email="kirkhamj@janelia.hhmi.org",
    url="https://github.com/nanshe-org/rank_filter",
    download_url="https://github.com/nanshe-org/rank_filter/archive/v%s.tar.gz"
                 % version,
    license="BSD",
    cmdclass=dict(
        list(versioneer.get_cmdclass().items()) +
        [
            ('build_ext', build_ext)
        ]
    ),
    setup_requires=setup_requires,
    build_requires=build_requires,
    install_requires=install_requires,
    tests_require=tests_require,
    test_suite="tests",
    headers=glob("include/*.hxx"),
    ext_modules=[Extension("rank_filter",
                            sources=sources,
                            include_dirs=include_dirs,
                            library_dirs=library_dirs,
                            libraries=libraries,
                            define_macros=define_macros,
                            extra_compile_args=extra_compile_args,
                            language="c++")],
    zip_safe=False
)
