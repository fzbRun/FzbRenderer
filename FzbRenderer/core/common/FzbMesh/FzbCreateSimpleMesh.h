#pragma once

#include "../FzbCommon.h"
#include "./FzbMesh.h"

#ifndef FZB_CREATE_SIMPLE_MESH_H
#define FZB_CREATE_SIMPLE_MESH_H

void fzbCreateCube(FzbMesh& mesh, FzbVertexFormat vertexFormat, glm::mat4 transformMatrix = glm::mat4(1.0f));

void fzbCreateCubeWireframe(FzbMesh& mesh, glm::mat4 transformMatrix = glm::mat4(1.0f));

void fzbCreateRectangle(FzbMesh& mesh, FzbVertexFormat vertexFormat, glm::mat4 transformMatrix = glm::mat4(1.0f));
#endif