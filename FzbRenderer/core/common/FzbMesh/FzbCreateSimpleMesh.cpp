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
        // ÿ�������ĸ�ԭʼ����������ɣ�����ԭ������˳��
        const int faces[6][4] = {
            {1, 0, 3, 2}, // z = 0 �棨back��
            {4, 5, 6, 7}, // z = 1 �棨front��
            {5, 1, 2, 6}, // x = 1 �棨right��
            {0, 4, 7, 3}, // x = 0 �棨left��
            {7, 6, 2, 3}, // y = 1 �棨top��
            {0, 1, 5, 4}  // y = 0 �棨bottom��
        };
        // ��Ӧÿ����ķ��ߣ������� faces ˳��һһ��Ӧ��
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

        // Ϊÿ���� push 4 ������
        for (int f = 0; f < 6; ++f) {
            for (int v = 0; v < 4; ++v) {
                int pi = faces[f][v];
                mesh.vertices.push_back(pos[pi][0]);    // λ��
                mesh.vertices.push_back(pos[pi][1]);
                mesh.vertices.push_back(pos[pi][2]);
                mesh.vertices.push_back(normals[f][0]);     // ���ߣ�����ķ��ߣ�
                mesh.vertices.push_back(normals[f][1]);
                mesh.vertices.push_back(normals[f][2]);
                if (vertexFormat.useTexCoord) {
                    mesh.vertices.push_back(texcoords[v][0]);   // texcoord
                    mesh.vertices.push_back(texcoords[v][1]);
                }
                if (vertexFormat.useTangent) throw std::runtime_error("ĿǰcreateCube����֧��tangent");
            }
            int base = f * 4;   // ÿ������������������ռ 4 �����㣬����������������
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
        if (vertexFormat.useTangent) {           // ����Ĭ�ϣ�����ָ�� +X��w=1��handedness��
            mesh.vertices[vertexStartIndex] = 1.0f;
            mesh.vertices[vertexStartIndex] = 0.0f;
            mesh.vertices[vertexStartIndex] = 0.0f;
            mesh.vertices[vertexStartIndex] = 1.0f;
        }
    }
    verticesTransform_CUDA(mesh.vertices, transformMatrix, mesh.vertexFormat);
}