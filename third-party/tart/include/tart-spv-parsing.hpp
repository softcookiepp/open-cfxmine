#ifndef TART_SPV_PARSER
#define TART_SPV_PARSER

#include <vector>
#include <iostream>
#include <string>
#include <stdexcept>
#include <spirv-tools/libspirv.hpp>

namespace tart
{

class SpvExtensionParser
{
private:
	// reference to spv. may not even need?
	const std::vector<uint32_t>& mSpv;
	
	// list of extension enums.
	std::vector<uint32_t> mExtensions;
	
	spv_result_t parseHeader(const spv_endianness_t endianess, const spv_parsed_header_t& instruction)
	{
		return SPV_SUCCESS;
	}
	
	spv_result_t parseInstruction(const spv_parsed_instruction_t& instruction)
	{
		if (instruction.opcode == SpvOpCapability)
		{
			for (size_t i = 0; i < instruction.num_operands; i += 1)
			{
				if (instruction.operands[i].type == SPV_OPERAND_TYPE_CAPABILITY)
				{
					auto& offset = instruction.operands[i].offset;
					auto& numWords = instruction.operands[i].num_words;
					for (size_t wordIdx = 0; wordIdx < numWords; wordIdx += 1)
					{
						mExtensions.push_back(instruction.words[wordIdx + offset]);
					}
				}
			}
		}
		return SPV_SUCCESS;
	}
	
public:
	SpvExtensionParser(const std::vector<uint32_t>& spv):
		mSpv(spv)
	{
		// valid spv must have length of at least 5
		if (spv.size() < 5) throw std::runtime_error("valid SPIR-V bytecode not supplied!");
		
		// why it gotta be like that...
		// https://stackoverflow.com/questions/23962019/how-to-initialize-stdfunction-with-a-member-function
		spvtools::HeaderParser headerParser([this](const spv_endianness_t endianess, const spv_parsed_header_t& instruction) { return this->parseHeader(endianess, instruction); } );
		spvtools::InstructionParser instructionParser([this](const spv_parsed_instruction_t& instruction) { return this->parseInstruction(instruction); } );
		spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_1);
		tools.Parse(spv, headerParser, instructionParser);
		
		// ok. we should have a complete list of the required extensions
		std::cout << "num capabilities: " << mExtensions.size() << std::endl;
	}
	~SpvExtensionParser()
	{
		
	}
};

} // namespace tart

#endif
