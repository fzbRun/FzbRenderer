#pragma once

#include <vector>
#include "../../FzbCommon.h"
#include "../../FzbMesh/FzbMesh.h"

#ifndef FZB_MESH_CUDA_H
#define FZB_MESH_CUDA_H

FzbMesh createVertices_CUDA(aiMesh* meshData, FzbVertexFormat vertexFormat);
void verticesTransform_CUDA(std::vector<float>& vertices, glm::mat4 transformMatrix, FzbVertexFormat vertexFormat);
std::vector<float> getVertices_CUDA(std::vector<float>& vertices, FzbVertexFormat vertexFormat, FzbVertexFormat vertexFormat_new);
FzbAABBBox createAABB_CUDA(std::vector<float>& vertices, FzbVertexFormat vertexFormat);

#endif