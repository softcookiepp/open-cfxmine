#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tart.hpp"
#include <optional>

namespace py = pybind11;

size_t getSizeFromBufferInfo(const py::buffer_info& info)
{
	size_t size = info.itemsize;
	for (size_t s : info.shape)
		size *= s;
	return size;
}

// convert python buffer into byte vector
std::vector<uint8_t> unpackPyBuffer(const py::buffer& buf)
{
	py::buffer_info info = buf.request();
	size_t size = getSizeFromBufferInfo(info);
	std::vector<uint8_t> out(size);
	std::memcpy(out.data(), info.ptr, size);
	return out;
}

std::vector<uint8_t> unpackOptionalPyBuffer(const std::optional<py::buffer>& buf)
{
	std::vector<uint8_t> unpacked;
	if (buf.has_value() )
		unpacked = unpackPyBuffer(buf.value());
	return unpacked;
}

PYBIND11_MODULE(pytart, m)
{
	m.doc() = "";
	
	py::class_<tart::Instance>(m, "Instance")
		.def(py::init<>() )
		.def("create_device", [](tart::Instance& self, uint32_t index, std::optional<std::vector<std::string>>& extensions)
			{
				std::vector<std::string> actualExtensions;
				if (extensions.has_value()) actualExtensions = extensions.value();
				return self.createDevice(index, actualExtensions);
			},
			py::arg("index"),
			py::arg("extensions") = py::none()
		);

	py::class_<tart::ShaderModule, tart::shader_module_ptr>(m, "ShaderModule")
		.def_property_readonly("spv", [](tart::shader_module_ptr self)
			{
				std::vector<uint32_t> spv = self->getSpv();
				return py::bytes((char*)(spv.data()), spv.size()*sizeof(uint32_t) );
			}
		);
	py::class_<tart::DeviceMetadata>(m, "DeviceMetadata")
		.def_property_readonly("char", [](tart::DeviceMetadata& self) { return self.char_; } )
		.def_property_readonly("uchar", [](tart::DeviceMetadata& self) { return self.uchar_; } )
		.def_property_readonly("short", [](tart::DeviceMetadata& self) { return self.short_; } )
		.def_property_readonly("ushort", [](tart::DeviceMetadata& self) { return self.ushort_; } )
		.def_property_readonly("int", [](tart::DeviceMetadata& self) { return self.int_; } )
		.def_property_readonly("uint", [](tart::DeviceMetadata& self) { return self.uint_; } )
		.def_property_readonly("long", [](tart::DeviceMetadata& self) { return self.long_; } )
		.def_property_readonly("ulong", [](tart::DeviceMetadata& self) { return self.ulong_; } )
		.def_property_readonly("fp8e4m3", [](tart::DeviceMetadata& self) { return self.fp8e4m3_; } )
		.def_property_readonly("fp8e5m2", [](tart::DeviceMetadata& self) { return self.fp8e5m2_; } )
		.def_property_readonly("half", [](tart::DeviceMetadata& self) { return self.half_; } )
		.def_property_readonly("float", [](tart::DeviceMetadata& self) { return self.float_; } )
		.def_property_readonly("double", [](tart::DeviceMetadata& self) { return self.double_; } )
		.def_property_readonly("bf16", [](tart::DeviceMetadata& self) { return self.bf16_; } );

	
	py::class_<tart::CLProgram, tart::cl_program_ptr>(m, "CLProgram")
		.def("dispatch", [](tart::cl_program_ptr self, std::string entryPoint, 
				std::vector<uint32_t> globalSize, std::vector<uint32_t> localSize, 
				std::vector<tart::buffer_ptr> buffers, std::optional<py::buffer> push_constants)
			{
				std::vector<uint8_t> pushConstants = unpackOptionalPyBuffer(push_constants);
				return self->dispatch(entryPoint, globalSize, localSize, buffers, pushConstants);
			},
			py::arg("entry_point"),
			py::arg("global_size"),
			py::arg("local_size"),
			py::arg("buffers"),
			py::arg("push_constants") = py::none()
		)
		.def("get_pipeline", [](tart::cl_program_ptr self,
				std::string entry_point, std::vector<uint32_t>& local_size,
				std::optional<py::buffer> push_constants)
			{
				std::vector<uint8_t> pushConstants = unpackOptionalPyBuffer(push_constants);
				return self->getPipeline(entry_point, local_size, pushConstants);
			},
			py::arg("entry_point"),
			py::arg("local_size"),
			py::arg("push_constants") = py::none()
		);
	
	py::class_<tart::Pipeline, tart::pipeline_ptr>(m, "Pipeline")
		.def("dispatch", [](tart::pipeline_ptr self,
				std::vector<uint32_t> workgroup,
				std::vector<tart::buffer_ptr> buffers,
				std::optional<py::buffer> push_constants)
			{
				std::vector<uint8_t> pushConstants = unpackOptionalPyBuffer(push_constants);
				return self->dispatch(workgroup, buffers, pushConstants);
			},
			py::arg("workgroup"),
			py::arg("buffers"),
			py::arg("push_constants") = py::none()
		)
		.def_property_readonly("num_buffer_args", [](tart::pipeline_ptr self) { return self->getNumBufferArguments(); } );
	
	py::class_<tart::CommandSequence, tart::command_sequence_ptr>(m, "Sequence")
		.def("record_copy_buffer", [](tart::command_sequence_ptr self, tart::buffer_ptr src, tart::buffer_ptr dst)
			{ return self->recordCopyBuffer(src, dst); } )
		.def("record_pipeline", [](
				tart::command_sequence_ptr self,
				tart::pipeline_ptr pipeline,
				std::vector<uint32_t> workgroup,
				std::vector<tart::buffer_ptr> buffers,
				std::optional<py::buffer> push_constants)
			{
				std::vector<uint8_t> pushConstants = unpackOptionalPyBuffer(push_constants);
				return self->recordPipeline(pipeline, workgroup, buffers, pushConstants);
			},
			py::arg("pipeline"),
			py::arg("workgroup"),
			py::arg("buffers"),
			py::arg("push_constants") = py::none()
		)
		.def_property_readonly("destroyed", [](tart::command_sequence_ptr self) { return self->isDestroyed(); } );
	
	py::class_<tart::Buffer, tart::buffer_ptr>(m, "Buffer")
		.def("copy_in", [](tart::buffer_ptr self, py::buffer& buf)
			{
				// first ensure size is correct
				py::buffer_info info = buf.request();
				size_t inpSize = getSizeFromBufferInfo(info);
				if (inpSize > self->getSize() - self->getOffset() )
					throw std::runtime_error("Size mismatch!");
				
				// then we are clear to copy
				self->copyIn(info.ptr, (uint32_t)inpSize);
			}
		)
		.def("copy_out", [](tart::buffer_ptr self, py::buffer& buf)
			{
				// first ensure size is correct
				py::buffer_info info = buf.request();
				size_t inpSize = getSizeFromBufferInfo(info);
				if (inpSize > self->getSize() - self->getOffset() )
					throw std::runtime_error("Size mismatch!");
				
				// then we are clear to copy
				self->copyOut(info.ptr, (uint32_t)inpSize);
			}
		)
		.def("view",[](tart::buffer_ptr self, uint32_t offset)
			{
				return self->view(offset);
			}
		)
		.def_property_readonly("address", [](tart::buffer_ptr self) { return self->getAddress(); })
		.def_property_readonly("size", [](tart::buffer_ptr self) { return self->getSize(); })
		.def_property_readonly("destroyed", [](tart::buffer_ptr self) { return self->isDestroyed(); })
		.def("copy_to", [](tart::buffer_ptr self, tart::buffer_ptr other, uint64_t self_offset, uint64_t dest_offset, uint64_t size)
			{ return self->copyTo(other, self_offset, dest_offset, size); },
			py::arg("other"),
			py::arg("self_offset") = 0,
			py::arg("dest_offset") = 0,
			py::arg("size") = 0
		)
		.def("copy_from", [](tart::buffer_ptr self, tart::buffer_ptr other)
			{ return self->copyFrom(other); } );
		
	py::class_<tart::Device, tart::device_ptr>(m, "Device")
		.def("is_destroyed", [](tart::device_ptr self)
			{ return self->isDestroyed(); }
		)
		/*
		 * TODO: add support for allocating with memory property flags.
		 * We have no idea whether or not the python peoples will want this capability.
		 */
		.def("allocate_buffer", [](tart::device_ptr self, uint64_t size, bool host)
			{ return self->allocateBuffer(size, host); },
			py::arg("size"),
			py::arg("host") = false
		)
		.def("allocate_buffer_from_array", [](tart::device_ptr self, uint32_t size, bool host)
			{
				// im too tired right now
				throw std::runtime_error("Not implemented!");
			}
		)
		.def_property_readonly("destroyed", [](tart::device_ptr self) { return self->isDestroyed(); } )
		.def("load_shader", [](tart::device_ptr self, const py::bytes& spirv )
			{
				py::buffer_info info(py::buffer(spirv).request());
				const char* data = reinterpret_cast<const char*>(info.ptr);
				size_t length = static_cast<size_t>(info.size);
				std::vector<uint32_t> spirvVec((uint32_t*)data, (uint32_t*)(data + length));
				return self->loadShader(spirvVec);
			}
		)
		.def("load_shader_from_path", [](tart::device_ptr self, py::str& path)
			{
				return self->loadShaderFromPath(path);
			}
		)
		.def("compile_glsl", [](tart::device_ptr self, py::str src) { return self->compileGLSL(src); })
		.def("compile_cl", [](tart::device_ptr self, std::string src) { tart::shader_module_ptr mod = self->compileCL(src); return mod; })
		.def("create_sequence", [](tart::device_ptr self) { return self->createSequence(); })
		.def("destroy_sequence", [](tart::device_ptr self, tart::command_sequence_ptr sequence)
			{ return self->destroySequence(sequence); } )
		.def("submit_sequence", [](tart::device_ptr self, tart::command_sequence_ptr sequence)
			{ return self->submitSequence(sequence); })
		.def("dispatch_pipeline", [](tart::device_ptr self, tart::pipeline_ptr pipeline, std::vector<uint32_t> workgroup, std::vector<tart::buffer_ptr> buffers)
			{ return self->dispatchPipeline(pipeline, workgroup, buffers); } )
		.def("sync", [](tart::device_ptr self) { return self->sync(); })
		.def("create_pipeline", [](tart::device_ptr self, tart::shader_module_ptr shader_module, std::string entry_point, std::optional<py::buffer> spec_constants, std::optional<py::buffer> push_constants)
			{
				std::vector<uint8_t> specConstants = unpackOptionalPyBuffer(spec_constants);
				std::vector<uint8_t> pushConstants = unpackOptionalPyBuffer(push_constants);
				return self->createPipeline(shader_module, entry_point, specConstants, pushConstants);
			},
			py::arg("shader_module"),
			py::arg("entry_point"),
			py::arg("spec_constants") = py::none(),
			py::arg("push_constants") = py::none()
		)
		.def("create_cl_program", [](tart::device_ptr self, tart::shader_module_ptr shader_module) { return self->createCLProgram(shader_module); } )
		.def("deallocate_buffer", [](tart::device_ptr self, tart::buffer_ptr buf) { return self->deallocateBuffer(buf); })
		.def_property_readonly("metadata", [](tart::device_ptr self) { return self->getMetadata(); });
}
