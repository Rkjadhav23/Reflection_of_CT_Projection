project(
    'OpenCLExercise1_Basics',
    ['cpp', 'c'],
    default_options : [
        'buildtype=debugoptimized',
        'optimization=3',
        'cpp_std=c++14',
        'warning_level=3',
    ],
)

cnpy_dep = dependency('cnpy', required: false)
zlib_dep = dependency('zlib', required: true)  # Add this line
if not cnpy_dep.found()
    cnpy_lib = static_library('cnpy', ['lib/cnpy.cpp'], include_directories: include_directories('lib'))
    cnpy_dep = declare_dependency(link_with: cnpy_lib, include_directories: include_directories('lib'))
endif
subdir('lib')

executable('OpenCLExercise1_Basics',
    [
        'src/OpenCLExercise1_Basics.cpp',
        xxd.process('src/OpenCLExercise1_Basics.cl'),
    ],
    extra_files: [
        'src/OpenCLExercise1_Basics.cl',
    ],
    dependencies : [
        core_dep,
        opencl_dep,
        cnpy_dep,
        zlib_dep
    ],
)
