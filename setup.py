from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os
import subprocess
import sysconfig

import pybind11

__version__ = '0.0.1'

try:
    __version__ += '-' + subprocess.check_output('git describe --always --long --dirty --abbrev=12'.split()).decode().strip()
except:
    pass

# build the shared object for fast vector lookup
dir = os.path.dirname(os.path.abspath(__file__))
# if not os.path.exists(os.path.join(dir, 'libfast_lookup.so')):
#     print('Run `make python`, to do everything or `make shared` to build the shared object first')
# # this is building the shared object in the setup? though that is probably not what we should be doing?
# c = 'cd {dir} && make MODE=optimized CUDA=0 shared'.format(dir=dir)
# print(c)
# r = os.system(c)
# assert r == 0


# this doesn't seem to work??????
class get_pybind_include:
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        try:
            import pybind11
            return pybind11.get_include(self.user)
        except ImportError:
            return ''

eec = ['-march=native', '-mtune=native', '-fno-stack-protector', '-funroll-loops', '-DNDEBUG', '-w', '-g1', '-DCERTIFIEDCOSINE_USE_PARALLEL', '-fopenmp']

extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
extra_compile_args = [e for e in extra_compile_args if ('mtune' not in e) and ('march' not in e) and ('stack' not in e)]
extra_compile_args += eec

ext_modules = [
    Extension(
        'certified_cosine',
        ['python_wrapper.cc'],
        include_dirs=[
            # Path to pybind11 headers
            # get_pybind_include(),
            # get_pybind_include(user=True)
            pybind11.get_include(True),
            pybind11.get_include(),
            dir,
            os.path.join(dir, '../eigen')
        ],
        runtime_library_dirs=[dir],
        library_dirs=[dir],
        extra_compile_args=extra_compile_args,
        extra_link_args=['-fopenmp'],
        language='c++'
    ),
]


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append('-std=c++17')  # we require c++17 for the lookup method, so new enough for pybind11
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args += opts
            # ext.extra_link_args.append('-Wl,-rpath=$ORIGIN')
            # ext.extra_link_args.append('fast_lookup.so')
        #import ipdb; ipdb.set_trace()
        build_ext.build_extensions(self)

setup(
    name='certified_cosine',
    version=__version__,
    author='Matthew Francis-Landau',
    author_email='matthew@matthewfl.com',
    url='https://github.com/matthewfl',
    description='Nearest neighbors with certificates',
    long_description='',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.2'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
