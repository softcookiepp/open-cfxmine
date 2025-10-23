#include <mutex>
#include <cstring>
#include <stdexcept>
#include "tart-compilers.hpp"

#ifdef ENABLE_LINKED_SHADER_COMPILERS
// headers for compiler libraries can be directly used
#include <glslang/Public/ShaderLang.h>
#include <glslang/Include/glslang_c_interface.h>
#include <glslang/Public/resource_limits_c.h>
#endif

// subprocess will allow the use of standalone compiler binaries if no compiler libaries are linked
#include <subprocess.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>

namespace tart_compilers
{
	

std::vector<uint32_t> runGlslangSubprocess(const std::string& src)
{
	// TODO: configure proper vulkan version and SPIR-V version
	
	// write source to temporary file
	std::filesystem::path srcPath("tmp.comp");
	if (std::filesystem::exists(srcPath))
		std::filesystem::remove(srcPath);
	std::ofstream srcStream;
	srcStream.open(srcPath);
	
	if (!srcStream.is_open())
		throw std::runtime_error("failed to open file!");
	
	srcStream << src;
	srcStream.close();
	
	if (!std::filesystem::exists(srcPath) )
		throw std::runtime_error("file could not be generated for some reason!");
	
	// set up output paths and stuff
	std::string dst("tmp.spv");
	std::filesystem::path dstPath(dst);
	if (std::filesystem::exists(dstPath))
		std::filesystem::remove(dstPath);
#if 0
	// invoke glslang
	std::string programName(subprocess::find_program("glslang") );
	subprocess::run({programName, "tmp.comp", "-S", "comp", "-V", "-g0", "-Os", "-o", dst, "--target-env", "vulkan1.3", "--quiet"});
#else
	std::string programName(subprocess::find_program("glslc") );
	subprocess::run({programName, "-fshader-stage=compute", "tmp.comp", "-O", "-Os", "-o", dst});
#endif
	// delete source
	//std::filesystem::remove(srcPath);
	
	// ensure spv exists, then read from path
	if (!std::filesystem::exists(dstPath))
		throw std::runtime_error("compile with glslang failed!");
	
	// read from out
	std::ifstream dstStream;
	dstStream.open(dstPath, std::ios::in | std::ios::binary | std::ios::ate);
	std::streampos size = dstStream.tellg();
	std::vector<uint32_t> out(size / sizeof(uint32_t));
	dstStream.seekg(0, std::ios::beg);
	dstStream.read( (char*)(out.data()), size);
	dstStream.close();
	
	// delete dst
	std::filesystem::remove(dstPath);
	
	// and we are done, I think!
	return out;
}	

std::vector<uint32_t> runCLSPVSubprocess(const std::string& src)
{
#if 1
	// TODO: configure proper vulkan version and SPIR-V version
	
	// write source to temporary file
	std::filesystem::path srcPath("tmp.cl");
	if (std::filesystem::exists(srcPath))
		std::filesystem::remove(srcPath);
	std::ofstream srcStream;
	srcStream.open(srcPath);
	
	if (!srcStream.is_open())
		throw std::runtime_error("failed to open file!");
	
	srcStream << src;
	srcStream.close();
	
	if (!std::filesystem::exists(srcPath) )
		throw std::runtime_error("file could not be generated for some reason!");
	
	// set up output paths and stuff
	std::string dst("tmp.spv");
	std::filesystem::path dstPath(dst);
	if (std::filesystem::exists(dstPath))
		std::filesystem::remove(dstPath);
	
	// invoke glslang
	std::string programName(subprocess::find_program("clspv") );
	subprocess::run({programName, "tmp.cl", "-o", dst, "-w"});
	
	// delete source
	//std::filesystem::remove(srcPath);
	
	// ensure spv exists, then read from path
	if (!std::filesystem::exists(dstPath))
		throw std::runtime_error("compile with clspv failed!");
	
	// read from out
	std::ifstream dstStream;
	dstStream.open(dstPath, std::ios::in | std::ios::binary | std::ios::ate);
	std::streampos size = dstStream.tellg();
	std::vector<uint32_t> out(size / sizeof(uint32_t));
	dstStream.seekg(0, std::ios::beg);
	dstStream.read( (char*)(out.data()), size);
	dstStream.close();
	
	// delete dst
	std::filesystem::remove(dstPath);
	
	// and we are done, I think!
	return out;
#else
	std::vector<uint32_t> spv;
	throw std::runtime_error("Not implemented!");
	return spv;
#endif
}


std::vector<uint32_t> compileCL(const std::string& src)
{
	#ifdef ENABLE_LINKED_SHADER_COMPILERS
	std::cerr << "WARNING: linking of clspv is not implemented yet due to dependency conflict issues" << std::endl; 
	#endif
	
	return runCLSPVSubprocess(src);
}


std::vector<uint32_t> compileGLSL(const std::string& src)
{
	std::vector<uint32_t> bin;
	#ifdef ENABLE_LINKED_SHADER_COMPILERS
	// we only do compute here, of course
	glslang_stage_t stage = GLSLANG_STAGE_COMPUTE;
	
	const std::string dummy("internal");
	const char* fileName = dummy.c_str();

	const glslang_input_t input = {
		.language = GLSLANG_SOURCE_GLSL,
		.stage = stage,
		.client = GLSLANG_CLIENT_VULKAN,
		.client_version = GLSLANG_TARGET_VULKAN_1_2,
		.target_language = GLSLANG_TARGET_SPV,
		.target_language_version = GLSLANG_TARGET_SPV_1_5,
		.code = src.c_str(),
		.default_version = 100,
		.default_profile = GLSLANG_NO_PROFILE,
		.force_default_version_and_profile = false,
		.forward_compatible = false,
		.messages = GLSLANG_MSG_DEFAULT_BIT,
		.resource = glslang_default_resource(),
	};

	glslang_shader_t* shader = glslang_shader_create(&input);
	
	if (!glslang_shader_preprocess(shader, &input))	{
		printf("GLSL preprocessing failed %s\n", fileName);
		printf("%s\n", glslang_shader_get_info_log(shader));
		printf("%s\n", glslang_shader_get_info_debug_log(shader));
		printf("%s\n", input.code);
		glslang_shader_delete(shader);
		return bin;
	}

	if (!glslang_shader_parse(shader, &input)) {
		printf("GLSL parsing failed %s\n", fileName);
		printf("%s\n", glslang_shader_get_info_log(shader));
		printf("%s\n", glslang_shader_get_info_debug_log(shader));
		printf("%s\n", glslang_shader_get_preprocessed_code(shader));
		glslang_shader_delete(shader);
		return bin;
	}

	glslang_program_t* program = glslang_program_create();
	glslang_program_add_shader(program, shader);

	if (!glslang_program_link(program, GLSLANG_MSG_SPV_RULES_BIT | GLSLANG_MSG_VULKAN_RULES_BIT)) {
		printf("GLSL linking failed %s\n", fileName);
		printf("%s\n", glslang_program_get_info_log(program));
		printf("%s\n", glslang_program_get_info_debug_log(program));
		glslang_program_delete(program);
		glslang_shader_delete(shader);
		return bin;
	}

	glslang_program_SPIRV_generate(program, stage);

	bin.resize(glslang_program_SPIRV_get_size(program));
	//bin.size = glslang_program_SPIRV_get_size(program);
	//bin.words = malloc(bin.size * sizeof(uint32_t));
	glslang_program_SPIRV_get(program, bin.data());

	const char* spirv_messages = glslang_program_SPIRV_get_messages(program);
	if (spirv_messages)
		printf("(%s) %s\b", fileName, spirv_messages);

	glslang_program_delete(program);
	glslang_shader_delete(shader);
	#else
	// not compiled with linked stuffs, just use subprocess lmao
	bin = runGlslangSubprocess(src);
	#endif
	return bin;
}

} // namespace tart_compilers
