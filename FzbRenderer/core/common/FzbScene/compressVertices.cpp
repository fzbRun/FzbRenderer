#include "compressVertices.h"
#include "FzbScene.h"
#include <thread>
#include <unordered_map>
#include <mutex>

static inline uint64_t hash_vertex(const float* floats, size_t vertexSize) noexcept {
    // FNV-1a 64-bit
    const uint64_t FNV_OFFSET_BASIS = 14695981039346656037ull;
    const uint64_t FNV_PRIME = 1099511628211ull;
    uint64_t hash = FNV_OFFSET_BASIS;

    for (size_t i = 0; i < vertexSize; ++i) {
        uint32_t bits;
        std::memcpy(&bits, &floats[i], sizeof(bits));

        // normalize ±0.0 -> +0.0 (all zero bits)
        if ((bits & 0x7FFFFFFFu) == 0u) bits = 0u;

        // normalize NaN -> canonical quiet NaN (0x7FC00000)
        if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0u) {
            bits = 0x7FC00000u;
        }

        // mix 4 bytes of bits into FNV-1a
        uint8_t b0 = static_cast<uint8_t>(bits & 0xFFu);
        uint8_t b1 = static_cast<uint8_t>((bits >> 8) & 0xFFu);
        uint8_t b2 = static_cast<uint8_t>((bits >> 16) & 0xFFu);
        uint8_t b3 = static_cast<uint8_t>((bits >> 24) & 0xFFu);

        hash ^= b0; hash *= FNV_PRIME;
        hash ^= b1; hash *= FNV_PRIME;
        hash ^= b2; hash *= FNV_PRIME;
        hash ^= b3; hash *= FNV_PRIME;
    }
    return hash;
}
struct VectorFloatHash {
    size_t operator()(std::vector<float> const& v) const noexcept {
        // 初始 seed（可以用长度，也可以换成常数）
        size_t seed = v.size();

        for (float f : v) {
            // 把 float 按位转换到 uint32_t（C++20 可用 std::bit_cast）
            uint32_t bits;
            std::memcpy(&bits, &f, sizeof(bits));

            // 规范化 ±0.0 -> 0
            if ((bits & 0x7FFFFFFFu) == 0u) {
                bits = 0u;
            }

            // 规范化 NaN（把任意 NaN 映射为一个 canonical NaN 比特形态）
            // IEEE-754 单精度：指数全1 且尾数非0 表示 NaN
            if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0u) {
                bits = 0x7FC00000u; // canonical quiet NaN
            }

            // 用 uint32_t 的哈希（或直接用 bits 参与更快的混合）
            size_t h = std::hash<uint32_t>{}(bits);

            // hash_combine 风格混合
            seed ^= h + static_cast<size_t>(0x9e3779b97f4a7c15ULL) + (seed << 6) + (seed >> 2);
        }

        return seed;
    };
};
struct Mat4Hash {
    size_t operator()(glm::mat4 const& m) const noexcept {
        // 把 16 个 float 串起来哈希
        size_t seed = 16;
        for (int col = 0; col < 4; ++col) {
            for (int row = 0; row < 4; ++row) {
                size_t h = std::hash<float>()(m[col][row]);
                seed ^= h + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
            }
        }
        return seed;
    };
};
struct Mat4Equal {
    bool operator()(glm::mat4 const& a, glm::mat4 const& b) const noexcept {
        // 如果你只想严格逐元素相等，可直接用 ==  
        // 但为了保险，也可以手动比：
        for (int col = 0; col < 4; ++col)
            for (int row = 0; row < 4; ++row)
                if (a[col][row] != b[col][row])
                    return false;
        return true;
    };
};

void compressSceneVertices(std::vector<float>& vertices, FzbVertexFormat vertexFormat, std::vector<uint32_t>& indices) {
	std::unordered_map<std::vector<float>, uint32_t, VectorFloatHash, std::equal_to<std::vector<float>>> uniqueVerticesMap{};
	std::vector<std::vector<float>> uniqueVertices;
	std::vector<uint32_t> uniqueIndices;

	uint32_t vertexSize = vertexFormat.getVertexSize();
	for (uint32_t i = 0; i < indices.size(); i++) {
		uint32_t vertexIndex = indices[i] * vertexSize;
		std::vector<float> vertex;
		vertex.insert(vertex.end(), vertices.begin() + vertexIndex, vertices.begin() + vertexIndex + vertexSize);

		if (uniqueVerticesMap.count(vertex) == 0) {
			uniqueVerticesMap[vertex] = static_cast<uint32_t>(uniqueVertices.size());
			uniqueVertices.push_back(vertex);
		}
		uniqueIndices.push_back(uniqueVerticesMap[vertex]);
	}

	vertices.clear();
	vertices.reserve(vertexSize * uniqueVertices.size());
	for (int i = 0; i < uniqueVertices.size(); i++) {
		vertices.insert(vertices.end(), uniqueVertices[i].begin(), uniqueVertices[i].end());
	}
	indices = uniqueIndices;
}
void compressSceneVertices_multiThread(std::vector<float>& vertices, FzbVertexFormat vertexFormat, std::vector<uint32_t>& indices) {
	uint32_t vertexSize = vertexFormat.getVertexSize();

	uint32_t threadNum = std::thread::hardware_concurrency();
	threadNum = std::max(1u, threadNum);
	threadNum = threadNum * 20 < indices.size() ? threadNum : std::ceil((float)indices.size() / 20);
	size_t indicesPerThread = (indices.size() + threadNum - 1) / threadNum;

	std::vector<std::unordered_map<std::vector<float>, uint32_t, VectorFloatHash, std::equal_to<std::vector<float>>>> localUniqueVerticesMap(threadNum);
	std::vector<std::vector<std::vector<float>>> localUniqueVertices(threadNum);
	std::vector<std::vector<uint32_t>> localUniqueIndices(threadNum);

	auto processChunk = [&](size_t threadIndex, size_t start, size_t end) {
		auto& uniqueVerticesMap = localUniqueVerticesMap[threadIndex];
		auto& uniqueVertices = localUniqueVertices[threadIndex];
		auto& uniqueIndices = localUniqueIndices[threadIndex];

		for (size_t i = start; i < end; ++i) {
			uint32_t vertexIndex = indices[i] * vertexSize;
			std::vector<float> vertex(vertices.begin() + vertexIndex, vertices.begin() + vertexIndex + vertexSize);
			if (uniqueVerticesMap.count(vertex) == 0) {
				uniqueVerticesMap[vertex] = static_cast<uint32_t>(uniqueVertices.size());
				uniqueVertices.push_back(vertex);
			}
			uniqueIndices.push_back(uniqueVerticesMap[vertex]);
		}
	};

	std::vector<std::thread> threads;
	for (uint32_t i = 0; i < threadNum; ++i) {
		size_t start = i * indicesPerThread;
		size_t end = std::min(start + indicesPerThread, indices.size());
		threads.emplace_back(processChunk, i, start, end);
	}
	for (auto& thread : threads) thread.join();

	std::unordered_map<std::vector<float>, uint32_t, VectorFloatHash, std::equal_to<std::vector<float>>> globalUniqueVerticesMap;
	std::vector<std::vector<float>> globalUniqueVertices;

	for (uint32_t i = 0; i < threadNum; ++i) {
		const auto& localVertices = localUniqueVertices[i];
		const auto& localIndices = localUniqueIndices[i];
		std::vector<uint32_t> indexMapping(localVertices.size());
		for (size_t j = 0; j < localVertices.size(); ++j) {
			const auto& vertex = localVertices[j];
			if (globalUniqueVerticesMap.count(vertex) == 0) {
				globalUniqueVerticesMap[vertex] = static_cast<uint32_t>(globalUniqueVertices.size());
				globalUniqueVertices.push_back(vertex);
			}
			indexMapping[j] = globalUniqueVerticesMap[vertex];
		}

		if (i == 0) {
			for (size_t j = 0; j < localIndices.size(); ++j) indices[j] = indexMapping[localIndices[j]];
		}
		else {
			size_t offset = i * indicesPerThread;
			for (size_t j = 0; j < localIndices.size(); ++j) indices[offset + j] = indexMapping[localIndices[j]];
		}
	}

	vertices.clear();
	vertices.reserve(vertexSize * globalUniqueVertices.size());
	for (const auto& vertex : globalUniqueVertices) vertices.insert(vertices.end(), vertex.begin(), vertex.end());
}
void compressSceneVertices_sharded(std::vector<float>& vertices, FzbVertexFormat vertexFormat, std::vector<uint32_t>& indices){
    const uint32_t vertexSize = static_cast<uint32_t>(vertexFormat.getVertexSize());
    const size_t indexCount = indices.size();
    if (indexCount == 0) return;

    unsigned threadNum = std::thread::hardware_concurrency();
    if (threadNum == 0) threadNum = 1;

    /*
    分片数大于线程数可以有效的避免线程访问到同一个分片，导致阻塞；很多测试表明2-8倍通常能获得最佳性能
    过多的分片也会需要更多的资源和管理成本
    */
    const unsigned shardFactor = 4;
    const unsigned shardCount = std::max<unsigned>(1, threadNum * shardFactor);

    struct Shard {
        std::mutex m;
        std::unordered_map<uint64_t, std::vector<uint32_t>> hashBuckets;
        std::vector<std::vector<float>> pool;   //hash值在该分片的顶点数据
    };
    std::vector<Shard> shards(shardCount);

    // 预分配内存
    size_t estUniqueTotal = std::min(indexCount, static_cast<size_t>(1000000)); // 保守上限
    size_t estPerShard = std::max<size_t>(4, estUniqueTotal / shardCount);  //假设一个分片处理多少个索引
    for (auto& s : shards) {
        s.pool.reserve(estPerShard);
        s.hashBuckets.reserve(std::max<size_t>(8, estPerShard / 4));    //假设一个hash对应4个索引
    }
    std::vector<uint64_t> encodedIndices(indexCount);   // 64位 shard/localIndex，表明索引和所处分片

    auto worker = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            uint32_t idx = indices[i];
            const float* vptr = vertices.data() + static_cast<size_t>(idx) * vertexSize;    //找到顶点数据

            uint64_t h = hash_vertex(vptr, vertexSize);
            unsigned sid = static_cast<unsigned>(h % shardCount);
            Shard& s = shards[sid];
            std::unique_lock<std::mutex> lk(s.m);

            auto it = s.hashBuckets.find(h);
            if (it == s.hashBuckets.end()) {
                // new hash bucket -> add pool entry
                uint32_t localIndex = static_cast<uint32_t>(s.pool.size());
                s.pool.emplace_back();
                s.pool.back().resize(vertexSize);
                std::memcpy(s.pool.back().data(), vptr, vertexSize * sizeof(float));
                s.hashBuckets[h] = std::vector<uint32_t>{ localIndex };
                encodedIndices[i] = (static_cast<uint64_t>(sid) << 32) | static_cast<uint64_t>(localIndex);
            }
            else {
                bool found = false;
                auto& vec = it->second;
                for (uint32_t localIdx : vec) { //如果hash值存在，则遍历相同hash的顶点，判断是否相等
                    if (std::memcmp(s.pool[localIdx].data(), vptr, vertexSize * sizeof(float)) == 0) {
                        encodedIndices[i] = (static_cast<uint64_t>(sid) << 32) | static_cast<uint64_t>(localIdx);
                        found = true;
                        break;
                    }
                }
                if (!found) {   //如果当前分片没有该顶点数据，那么在pool中添加该数据；并将在pool中的索引放入hash桶相应hash的vector中
                    uint32_t localIndex = static_cast<uint32_t>(s.pool.size());
                    s.pool.emplace_back();
                    s.pool.back().resize(vertexSize);
                    std::memcpy(s.pool.back().data(), vptr, vertexSize * sizeof(float));
                    vec.push_back(localIndex);
                    encodedIndices[i] = (static_cast<uint64_t>(sid) << 32) | static_cast<uint64_t>(localIndex);
                }
            }
            // unlock when lk goes out of scope
        }
    };

    // spawn threads
    std::vector<std::thread> threads;
    threads.reserve(threadNum);
    size_t chunk = (indexCount + threadNum - 1) / threadNum;
    for (unsigned t = 0; t < threadNum; ++t) {
        size_t s = t * chunk;
        size_t e = std::min(indexCount, s + chunk);
        if (s >= e) break;
        threads.emplace_back(worker, s, e);
    }
    for (auto& th : threads) th.join();

    std::vector<float> newVertices;
    newVertices.reserve(vertexSize * indexCount);

    /*
    现在我们有若干个分片，每个分片的pool存有顶点数据（hash桶已经没用了，这里）
    encodedIndices存有每个索引对应的顶点数据在哪个分片和在分片pool中的索引
    那么，现在我们先将所有分片的顶点数据放入全局数组中，然后记录存入全局数组的索引到remap中
    然后，每个索引将对应分片的对应pool中的位置作为索引，找到remap中的索引，即可
    */
    std::vector<std::vector<uint32_t>> remap(shardCount);   //每个分片的pool中的顶点数据在全局数组中的索引
    for (unsigned sid = 0; sid < shardCount; ++sid) {   //遍历每个分片的pool，将
        Shard& s = shards[sid];
        remap[sid].resize(s.pool.size());
        for (size_t localIdx = 0; localIdx < s.pool.size(); ++localIdx) {
            uint32_t globalIndex = static_cast<uint32_t>(newVertices.size() / vertexSize);
            remap[sid][localIdx] = globalIndex;
            // append vertex data
            const std::vector<float>& v = s.pool[localIdx];
            newVertices.insert(newVertices.end(), v.begin(), v.end());
        }
    }

    std::vector<uint32_t> finalIndices(indexCount);
    for (size_t i = 0; i < indexCount; ++i) {
        uint64_t enc = encodedIndices[i];
        unsigned sid = static_cast<unsigned>(enc >> 32);
        uint32_t localIdx = static_cast<uint32_t>(enc & 0xFFFFFFFFu);
        finalIndices[i] = remap[sid][localIdx];
    }

    indices.swap(finalIndices);
    vertices.swap(newVertices);
}