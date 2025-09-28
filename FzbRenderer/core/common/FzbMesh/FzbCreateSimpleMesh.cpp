#include "./FzbCreateSimpleMesh.h"
#include "CUDA/FzbMeshCUDA.cuh"

void fzbCreateCube(FzbMesh& mesh, FzbVertexFormat vertexFormat, glm::mat4 transformMatrix) {
    mesh.vertexFormat = vertexFormat;
    if (vertexFormat == FzbVertexFormat()) {
        mesh.vertices = { 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f };
        mesh.indices = {
                        1, 0, 3, 1, 3, 2,
                        4, 5, 6, 4, 6, 7,
                        5, 1, 2, 5, 2, 6,
                        0, 4, 7, 0, 7, 3,
                        7, 6, 2, 7, 2, 3,
                        0, 1, 5, 0, 5, 4
        };
    }
    else {
        const float pos[8][3] = {
                {0.0f, 0.0f, 0.0f}, // 0
                {1.0f, 0.0f, 0.0f}, // 1
                {1.0f, 1.0f, 0.0f}, // 2
                {0.0f, 1.0f, 0.0f}, // 3
                {0.0f, 0.0f, 1.0f}, // 4
                {1.0f, 0.0f, 1.0f}, // 5
                {1.0f, 1.0f, 1.0f}, // 6
                {0.0f, 1.0f, 1.0f}  // 7
        };
        // 每个面由四个原始顶点索引组成（按你原来的面顺序）
        const int faces[6][4] = {
            {1, 0, 3, 2}, // z = 0 面（back）
            {4, 5, 6, 7}, // z = 1 面（front）
            {5, 1, 2, 6}, // x = 1 面（right）
            {0, 4, 7, 3}, // x = 0 面（left）
            {7, 6, 2, 3}, // y = 1 面（top）
            {0, 1, 5, 4}  // y = 0 面（bottom）
        };
        // 对应每个面的法线（与上面 faces 顺序一一对应）
        const float normals[6][3] = {
            { 0.0f,  0.0f, -1.0f}, // back
            { 0.0f,  0.0f,  1.0f}, // front
            { 1.0f,  0.0f,  0.0f}, // right
            {-1.0f,  0.0f,  0.0f}, // left
            { 0.0f,  1.0f,  0.0f}, // top
            { 0.0f, -1.0f,  0.0f}  // bottom
        };
        const float texcoords[4][2] = {
            {0.0f, 0.0f},
            {1.0f, 0.0f},
            {1.0f, 1.0f},
            {0.0f, 1.0f}
        };
        mesh.vertices.clear();
        mesh.indices.clear();

        // 为每个面 push 4 个顶点
        for (int f = 0; f < 6; ++f) {
            for (int v = 0; v < 4; ++v) {
                int pi = faces[f][v];
                mesh.vertices.push_back(pos[pi][0]);    // 位置
                mesh.vertices.push_back(pos[pi][1]);
                mesh.vertices.push_back(pos[pi][2]);
                mesh.vertices.push_back(normals[f][0]);     // 法线（该面的法线）
                mesh.vertices.push_back(normals[f][1]);
                mesh.vertices.push_back(normals[f][2]);
                if (vertexFormat.useTexCoord) {
                    mesh.vertices.push_back(texcoords[v][0]);   // texcoord
                    mesh.vertices.push_back(texcoords[v][1]);
                }
                if (vertexFormat.useTangent) throw std::runtime_error("目前createCube还不支持tangent");
            }
            int base = f * 4;   // 每个面在我们新数组里占 4 个顶点，按两个三角形索引
            mesh.indices.push_back(base + 0);
            mesh.indices.push_back(base + 1);
            mesh.indices.push_back(base + 2);

            mesh.indices.push_back(base + 0);
            mesh.indices.push_back(base + 2);
            mesh.indices.push_back(base + 3);
        }
    }
    mesh.indexArraySize = mesh.indices.size();
    verticesTransform_CUDA(mesh.vertices, transformMatrix, mesh.vertexFormat);
}

void fzbCreateCubeWireframe(FzbMesh& mesh, glm::mat4 transformMatrix) {
    mesh.vertices = { 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                      0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f };
    mesh.indices = {
        0, 1, 1, 2, 2, 3, 3, 0,
        4, 5, 5, 6, 6, 7, 7, 4,
        0, 4, 1, 5, 2, 6, 3, 7
    };
    mesh.indexArraySize = mesh.indices.size();
    mesh.vertexFormat = FzbVertexFormat();
    verticesTransform_CUDA(mesh.vertices, transformMatrix, mesh.vertexFormat);
}

void fzbCreateRectangle(FzbMesh& mesh, FzbVertexFormat vertexFormat, glm::mat4 transformMatrix) {
    mesh.indices = {
        0, 1, 2,
        0, 2, 3
    };
    mesh.indexArraySize = mesh.indices.size();
    mesh.vertexFormat = vertexFormat;

    const float pos[4][3] = {
                {0.0f, 0.0f, 0.0f}, // 0
                {1.0f, 0.0f, 0.0f}, // 1
                {1.0f, 1.0f, 0.0f}, // 2
                {0.0f, 1.0f, 0.0f}, // 3
    };
    const float normals[3] = { 0.0f,  0.0f,  1.0f };
    const float texcoords[4][2] = {
        {0.0f, 0.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f},
        {0.0f, 1.0f}
    };

    uint32_t vertexSize = vertexFormat.getVertexSize();
    mesh.vertices.resize(vertexSize * 4);

    for (int i = 0; i < 4; ++i) {
        uint32_t vertexIndex = i;
        uint32_t vertexStartIndex = vertexIndex * vertexSize;
        mesh.vertices[vertexStartIndex++] = pos[vertexIndex][0];
        mesh.vertices[vertexStartIndex++] = pos[vertexIndex][1];
        mesh.vertices[vertexStartIndex++] = pos[vertexIndex][2];
        if (vertexFormat.useNormal) {
            mesh.vertices[vertexStartIndex++] = normals[0];
            mesh.vertices[vertexStartIndex++] = normals[1];
            mesh.vertices[vertexStartIndex++] = normals[2];
        }
        if (vertexFormat.useTexCoord) {
            mesh.vertices[vertexStartIndex++] = texcoords[vertexIndex][0];
            mesh.vertices[vertexStartIndex++] = texcoords[vertexIndex][1];
        }
        if (vertexFormat.useTangent) {           // 常见默认：切线指向 +X，w=1（handedness）
            mesh.vertices[vertexStartIndex] = 1.0f;
            mesh.vertices[vertexStartIndex] = 0.0f;
            mesh.vertices[vertexStartIndex] = 0.0f;
            mesh.vertices[vertexStartIndex] = 1.0f;
        }
    }
    verticesTransform_CUDA(mesh.vertices, transformMatrix, mesh.vertexFormat);
}