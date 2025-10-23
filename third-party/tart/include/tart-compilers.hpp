#ifndef TART_COMPILERS
#define TART_COMPILERS
#include <vector>
#include <string>

namespace tart_compilers
{

std::vector<uint32_t> compileCL(const std::string& src);

std::vector<uint32_t> compileGLSL(const std::string& src);

} // namespace tart_compilers
#endif
