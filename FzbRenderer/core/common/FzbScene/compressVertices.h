#pragma once
#include <vector>
#include "../StructSet.h"

#ifndef COMPRESS_VERTICES_H
#define COMPRESS_VERTICES_H

void compressSceneVertices(std::vector<float>& vertices, FzbVertexFormat vertexFormat, std::vector<uint32_t>& indices);
void compressSceneVertices_multiThread(std::vector<float>& vertices, FzbVertexFormat vertexFormat, std::vector<uint32_t>& indices);
void compressSceneVertices_sharded(std::vector<float>& vertices, FzbVertexFormat vertexFormat, std::vector<uint32_t>& indices);

#endif