# Tart

Tart is a GPGPU framework written in Vulkan.
It is designed to reduce the required boilerplate for harnessing Vulkan's compute capabilities,
while still not compromising in flexibility or feature set.

## Building

Regardless of interface, you will need gcc, g++, python3 headers, a vulkan driver, and CMake.
On Ubuntu and other Debian-based distros, these can be installed by running:
```bash
sudo apt install build-essential python3-dev 
```

The vulkan driver will depend on your device, but most should be accessible through mesa:
```bash
sudo apt install mesa-vulkan-drivers
```

For NVIDIA GPUs, driver install will depend on your GPU architecture. I do not yet have instructions for that.

### Python bindings
To install the python bindings, ensure GCC and CMake are installed.
Then do the following to install and run the python unit tests.:
```bash
git clone https://codeberg.org/softcookiepp/tart.git --recurse-submodules
pip install ".[test]"
```

### C++ library
To build tart with no statically linked compilers:
```
mkdir build
cd build
cmake ..
make
```

To build tart with compilers included (currently only glslang):
```
mkdir build
cd build
cmake -DTART_ENABLE_SHADER_COMPILERS=1 -DCMAKE_BUILD_TYPE=Release
make
```

## Testing
Unit tests are built along with everything else by default.
Simply execute the following to run them:
```
./run-tests
```

### Roadmap
- implement enabling of some default extensions for capabilities that are commonly used (partially done)
- make cross-platform, easily installable python bindings (likely will use https://scikit-build-core.readthedocs.io/en/latest/ and https://pybind11.readthedocs.io/en/stable/compiling.html) (in-progress)
- make python bindings for clspv (probably will be a separate project altogether)
- fix bugs and stuff
