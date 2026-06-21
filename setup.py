from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

import numpy as np


class build_ext_openmp(build_ext):
    """Add OpenMP + optimization flags appropriate to the active compiler.

    OpenMP is used only for the embarrassingly parallel candidate-feature split
    search; the parallel reduction is deterministic, so results stay identical
    to the single-threaded build.  The pragmas are guarded by ``_OPENMP`` in the
    C source, so a build without these flags still compiles and runs serially.
    """

    def build_extensions(self):
        ct = self.compiler.compiler_type
        if ct == "msvc":
            extra_compile = ["/openmp"]
            extra_link = []
        else:  # gcc / clang
            extra_compile = ["-fopenmp", "-O3"]
            extra_link = ["-fopenmp"]
        for ext in self.extensions:
            ext.extra_compile_args = (ext.extra_compile_args or []) + extra_compile
            ext.extra_link_args = (ext.extra_link_args or []) + extra_link
        super().build_extensions()


setup(
    cmdclass={"build_ext": build_ext_openmp},
    ext_modules=[
        Extension(
            "fuzzydtree._c_backend",
            ["fuzzydtree/_c_backend.c"],
            include_dirs=[np.get_include()],
        )
    ],
)
