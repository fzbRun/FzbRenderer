#include "FzbCamera.h"

FzbCamera::FzbCamera(glm::vec3 position, glm::vec3 up, float yaw, float pitch) : Front(glm::vec3(0.0f, 0.0f, -1.0f)), MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY), Zoom(ZOOM)
{
    position = position;
    WorldUp = up;
    Yaw = yaw;
    Pitch = pitch;
    updateCameraVectors();
}

FzbCamera::FzbCamera(float posX, float posY, float posZ, float upX, float upY, float upZ, float yaw, float pitch) : Front(glm::vec3(0.0f, 0.0f, -1.0f)), MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY), Zoom(ZOOM)
{
    position = glm::vec3(posX, posY, posZ);
    WorldUp = glm::vec3(upX, upY, upZ);
    Yaw = yaw;
    Pitch = pitch;
    updateCameraVectors();
}

FzbCamera::FzbCamera(glm::vec3 position, float fov, float aspect, float nearPlane, float farPlane, glm::vec3 up, float yaw, float pitch) : Front(glm::vec3(0.0f, 0.0f, -1.0f)), MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY), Zoom(ZOOM) {
    this->position = position;
    this->WorldUp = up;
    this->Yaw = yaw;
    this->Pitch = pitch;
    updateCameraVectors();

    this->fov = fov;
    this->aspect = aspect;
    this->nearPlane = nearPlane;
    this->farPlane = farPlane;
}

void FzbCamera::setViewMatrix(glm::mat4 viewMatrix, bool inverse) {
    glm::mat4 inverseMatrix;
    if (inverse) {
        inverseMatrix = viewMatrix;
        this->viewMatrix = glm::inverse(viewMatrix);
    }
    else {
        this->viewMatrix = viewMatrix;
        inverseMatrix = glm::inverse(viewMatrix);
    }
    setFront(inverseMatrix[2]);
    //this->Front = inverseMatrix[2];
    //this->Up = inverseMatrix[1];
    //
    //this->Pitch = glm::degrees(glm::asin(Front.y));
    //this->Yaw = glm::degrees(glm::acos(Front.x - glm::sqrt(1.0f - Front.y * Front.y)));
}

void FzbCamera::createViewMatrix() {
    this->viewMatrix = glm::lookAt(position, position + Front, Up);
}
glm::mat4 FzbCamera::GetViewMatrix()
{
    createViewMatrix();
    return this->viewMatrix;
}

void FzbCamera::createProjMatrix() {
    if (this->isPerspective) {
        this->projMatrix = glm::perspectiveRH_ZO(this->fov, this->aspect, this->nearPlane, this->farPlane);
        this->projMatrix[1][1] *= -1;
    }
    else {
        this->projMatrix = glm::orthoRH_ZO(-10.0f, 10.0f, -10.0f, 10.0f, this->nearPlane, this->farPlane);
    }
}
glm::mat4 FzbCamera::GetProjMatrix() {
    if (this->projMatrix == glm::mat4(0.0f)) createProjMatrix();
    return this->projMatrix;
}

void FzbCamera::ProcessKeyboard(Camera_Movement direction, float deltaTime)
{
    float velocity = MovementSpeed * deltaTime;
    if (direction == FORWARD)
        position += Front * velocity;
    if (direction == BACKWARD)
        position -= Front * velocity;
    if (direction == LEFT)
        position -= Right * velocity;
    if (direction == RIGHT)
        position += Right * velocity;
}

void FzbCamera::ProcessMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch)
{
    xoffset *= MouseSensitivity * 2;
    yoffset *= MouseSensitivity * 2;

    Yaw += xoffset;
    Pitch += yoffset;

    // make sure that when pitch is out of bounds, screen doesn't get flipped
    if (constrainPitch)
    {
        if (Pitch > 89.0f)
            Pitch = 89.0f;
        if (Pitch < -89.0f)
            Pitch = -89.0f;
    }

    // update Front, Right and Up Vectors using the updated Euler angles
    updateCameraVectors();
}

void FzbCamera::ProcessMouseScroll(float yoffset)
{
    Zoom -= (float)yoffset;
    if (Zoom < 1.0f)
        Zoom = 1.0f;
    if (Zoom > 45.0f)
        Zoom = 45.0f;
}

void FzbCamera::updateCameraVectors()
{
    // calculate the new Front vector
    glm::vec3 front;
    front.x = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch));
    front.y = sin(glm::radians(Pitch));
    front.z = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch));
    Front = glm::normalize(front);
    // also re-calculate the Right and Up vector
    Right = glm::normalize(glm::cross(Front, WorldUp));  // normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
    Up = glm::normalize(glm::cross(Right, Front));
}

void FzbCamera::setFront(const glm::vec3& front)
{
    // 规范化输入向量
    glm::vec3 normalizedFront = glm::normalize(front);

    // 计算 Pitch (俯仰角)
    Pitch = glm::degrees(asin(normalizedFront.y));

    // 计算 Yaw (偏航角)
    Yaw = glm::degrees(atan2(normalizedFront.z, normalizedFront.x));

    // 确保角度在合理范围内
    if (Pitch > 89.0f)
        Pitch = 89.0f;
    if (Pitch < -89.0f)
        Pitch = -89.0f;

    // 更新相机向量
    updateCameraVectors();
}