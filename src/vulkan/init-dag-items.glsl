#version 450
#extension GL_EXT_buffer_reference : require

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
layout(constant_id = 0) uint LX = 1; // TODO: set these to the defaults that the CUDA miner uses
layout(constant_id = 1) uint LY = 1;
layout(constant_id = 2) uint LZ = 1;

struct octopus_h256_t {
  uint8_t b[32];
};

struct octopus_return_value_t {
  octopus_h256_t result;
  bool success;
};

#define MAX_SEARCH_RESULTS 4

struct SearchResult {
  uint32_t nonce_offset;
  uint32_t pad[1];
};

struct SearchResults {
  SearchResult result[MAX_SEARCH_RESULTS];
  uint32_t count = 0;
};

#if 1
struct hash32_t
{
	// this will be dumb, but whatever
	uvec2 uint2s[16];
};
#else
typedef union {
  uint2 uint2s[32 / sizeof(uint2)];
  uint4 uint4s[32 / sizeof(uint4)];
} hash32_t;
#endif

#if 1
// so how does this work?
// I will basically have to have macros that convert this correctly...
struct hash64_t {
  uint words[16];
  //uint2 uint2s[64 / sizeof(uint2)];
  //uint4 uint4s[64 / sizeof(uint4)];
};
#else
typedef union {
  uint32_t words[64 / sizeof(uint32_t)];
  uint2 uint2s[64 / sizeof(uint2)];
  uint4 uint4s[64 / sizeof(uint4)];
} hash64_t;
#endif

#if 1
struct hash256_t{
	uvec4 uint4s[16];
};
#else
typedef struct {
  uint4 uint4s[256 / sizeof(uint4)];
} hash256_t;
#endif

#if 1
struct hash200_t {
  uint words[50];
  //uint2 uint2s[200 / sizeof(uint2)];
  //uint4 uint4s[200 / sizeof(uint4)];
};
#else
typedef union {
  uint32_t words[200 / sizeof(uint32_t)];
  uint2 uint2s[200 / sizeof(uint2)];
  uint4 uint4s[200 / sizeof(uint4)];
} hash200_t;
#endif

layout(std430, buffer_reference, buffer_reference_align = 8) buffer hash64_buf
{
    hash64_t data[];
};

layout(push_constant) uniform push_consts
{
	hash64_buf d_light;
	uint start;
} k;

void main()
{
#if 1
	const uint node_inde = k.start + gl_WorkGroupID.x*LX + gl_LocalInvocationID.x;
#else
	const uint32_t node_index = k.start + blockIdx.x * blockDim.x + threadIdx.x;
#endif
	if ((node_index & ~3) >= d_dag_size * MIX_NODES) {
		return;
	}

	hash200_t dag_node;
	#pragma unroll
	for (int i = 0; i < 4; ++i) {
	dag_node.uint4s[i] = d_light.data[node_index % d_light_size].uint4s[i];
	}
	dag_node.words[0] ^= node_index;
	SHA3_512(dag_node.uint2s);

	const int thread_id = threadIdx.x & 3;

	for (uint32_t i = 0; i != OCTOPUS_DATASET_PARENTS; ++i) {
	uint32_t parent_index =
	fnv(node_index ^ i, dag_node.words[i % NODE_WORDS]) % d_light_size;
	for (uint32_t t = 0; t < 4; t++) {
	uint32_t shuffle_index = SHFL(parent_index, t, 4);
	uint4 p4 = d_light[shuffle_index].uint4s[thread_id];
	for (int w = 0; w < 4; w++) {
	uint4 s4 = make_uint4(SHFL(p4.x, w, 4), SHFL(p4.y, w, 4),
						  SHFL(p4.z, w, 4), SHFL(p4.w, w, 4));
	if (t == thread_id) {
	  dag_node.uint4s[w] = fnv4(dag_node.uint4s[w], s4);
	}
	}
	}
	}
	SHA3_512(dag_node.uint2s);
	hash64_t *dag_nodes = (hash64_t *)d_dag;

	for (uint32_t t = 0; t < 4; t++) {
	uint32_t shuffle_index = SHFL(node_index, t, 4);
	uint4 s[4];
	for (uint32_t w = 0; w < 4; w++) {
	s[w] = make_uint4(
	  SHFL(dag_node.uint4s[w].x, t, 4), SHFL(dag_node.uint4s[w].y, t, 4),
	  SHFL(dag_node.uint4s[w].z, t, 4), SHFL(dag_node.uint4s[w].w, t, 4));
	}
	if (shuffle_index < d_dag_size * MIX_NODES) {
	dag_nodes[shuffle_index].uint4s[thread_id] = s[thread_id];
	}
	}
}
