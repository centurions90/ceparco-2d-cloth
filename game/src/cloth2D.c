#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "raylib.h"
#include "raymath.h"

//----------------------------------------------------------------------------------
// Variables Definition
//----------------------------------------------------------------------------------
static const int screenWidth = 1920;
static const int screenHeight = 1080;
static int N;                           // Cloth resolution
static float size = 10.0f;
static float spacing;
static int iterations;
static float gravity = 1.0f / 60.0f;
static double totalTime = 0;

typedef struct Cloth
{
    Vector3* vertices;
    Vector3* prevVertices;
} Cloth;

typedef struct ClothMesh
{
    Mesh mesh;
    int** indices;
} ClothMesh;

//----------------------------------------------------------------------------------
// Functions Declaration
//----------------------------------------------------------------------------------

static void UpdateDrawFrame(Camera* camera, Model* cloth);  // Update and draw one frame
static Cloth CreateCloth(int N);                            // Create cloth with an NxN resolution
static Mesh GenClothMesh(Cloth* cloth);                     // Create mesh to render cloth
static void UpdateClothMesh(Mesh* mesh, Cloth* cloth);      // Update cloth mesh to match data
static void UpdateCloth(Cloth* cloth, Camera* camera);                      // Update cloth physics
static void LinkConstraint(Cloth* cloth, int p1, int p2);

//----------------------------------------------------------------------------------
// Main entry point
//----------------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    // Initialization
    //---------------------------------------------------------
    InitWindow(screenWidth, screenHeight, "Cloth Simulation");
    SetTargetFPS(60);

    if (argc >= 2)
    {
        N = atoi(argv[1]);
    }
    else
    {
        N = 10;
    }

    if (argc >= 3)
    {
        iterations = atoi(argv[2]);
    }
    else
    {
        iterations = 30;
    }

    Cloth cloth = CreateCloth(N);
    Mesh clothMesh = GenClothMesh(&cloth);
    Model clothModel = LoadModelFromMesh(clothMesh);

    Camera camera = { { 0.0f, 0.0f, 5.0f }, { 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, 15.0f, CAMERA_ORTHOGRAPHIC };

    clock_t start, end;
    double timeTaken;
    int count = 0;

    // Main game loop
    while (!WindowShouldClose())    // Detect window close button or ESC key
    {
        start = clock();
        UpdateCloth(&cloth, &camera);
        end = clock();
        timeTaken = (double)(end - start) * 1e6 / CLOCKS_PER_SEC;
        totalTime += timeTaken;
        count++;
        UpdateClothMesh(&clothMesh, &cloth);
        UpdateDrawFrame(&camera, &clothModel);
    }

    // De-Initialization
    //--------------------------------------------------------------------------------------

    CloseWindow();          // Close window and OpenGL context
    //--------------------------------------------------------------------------------------
    printf("--------------------------------------------\n");
    printf("Average physics execution time took %f microseconds\n", totalTime / count);

    return 0;
}

//----------------------------------------------------------------------------------
// Module specific Functions Definition
//----------------------------------------------------------------------------------
// Update and draw game frame
static void UpdateDrawFrame(Camera* camera, Model* cloth)
{
    // Draw
    //----------------------------------------------------------------------------------
    BeginDrawing();

    ClearBackground(BLACK);

    BeginMode3D(*camera);

    DrawModelWires(*cloth, Vector3Zero(), 1.0f, RED);

    EndMode3D();

    DrawFPS(30, 30);

    EndDrawing();
    //----------------------------------------------------------------------------------
}

static Cloth CreateCloth(int N)
{
    float originX = -size / 2.0f;
    float originY = size / 2.0f;
    spacing = 1.0f / N * size;

    Cloth cloth = { 0 };
    cloth.vertices = (Vector3*)MemAlloc(N * N * sizeof(Vector3));
    cloth.prevVertices = (Vector3*)MemAlloc(N * N * sizeof(Vector3));

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int index = i * N + j;

            cloth.vertices[index] = (Vector3) {
                originX + spacing * j,
                originY - spacing * i,
                0
            };

            cloth.prevVertices[index] = cloth.vertices[index];
        }
    }

    return cloth;
}

static Mesh GenClothMesh(Cloth* cloth)
{
    Mesh clothMesh = { 0 };
    clothMesh.triangleCount = (N - 1) * (N - 1) * 2;
    clothMesh.vertexCount = clothMesh.triangleCount * 3;
    clothMesh.vertices = (float*)MemAlloc(clothMesh.vertexCount * 3 * sizeof(float));    // 3 vertices, 3 coordinates each (x, y, z)
    clothMesh.texcoords = (float*)MemAlloc(clothMesh.vertexCount * 2 * sizeof(float));   // 3 vertices, 2 coordinates each (x, y)
    clothMesh.normals = (float*)MemAlloc(clothMesh.vertexCount * 3 * sizeof(float));     // 3 vertices, 3 coordinates each (x, y, z)

    for (int i = 0; i < N - 1; i++)
    {
        for (int j = 0; j < N - 1; j++)
        {
            int index = i * (N - 1) + j;
            int vertex = i * (N - 1) + j + i;

            clothMesh.vertices[index * 9] = cloth->vertices[vertex].x;
            clothMesh.vertices[index * 9 + 1] = cloth->vertices[vertex].y;
            clothMesh.vertices[index * 9 + 2] = cloth->vertices[vertex].z;
            clothMesh.normals[index * 9] = 0;
            clothMesh.normals[index * 9 + 1] = 0;
            clothMesh.normals[index * 9 + 2] = 1;
            clothMesh.texcoords[index * 6] = 0;
            clothMesh.texcoords[index * 6 + 1] = 0;

            clothMesh.vertices[index * 9 + 3] = cloth->vertices[vertex + N].x;
            clothMesh.vertices[index * 9 + 4] = cloth->vertices[vertex + N].y;
            clothMesh.vertices[index * 9 + 5] = cloth->vertices[vertex + N].z;
            clothMesh.normals[index * 9 + 3] = 0;
            clothMesh.normals[index * 9 + 4] = 0;
            clothMesh.normals[index * 9 + 5] = 1;
            clothMesh.texcoords[index * 6 + 2] = 0;
            clothMesh.texcoords[index * 6 + 3] = 0;

            clothMesh.vertices[index * 9 + 6] = cloth->vertices[vertex + 1].x;
            clothMesh.vertices[index * 9 + 7] = cloth->vertices[vertex + 1].y;
            clothMesh.vertices[index * 9 + 8] = cloth->vertices[vertex + 1].z;
            clothMesh.normals[index * 9 + 6] = 0;
            clothMesh.normals[index * 9 + 7] = 0;
            clothMesh.normals[index * 9 + 8] = 1;
            clothMesh.texcoords[index * 6 + 4] = 0;
            clothMesh.texcoords[index * 6 + 5] = 0;
        }
    }

    for (int i = 0; i < N - 1; i++)
    {
        for (int j = 0; j < N - 1; j++)
        {
            int index = i * (N - 1) + j + (N - 1) * (N - 1);
            int vertex = i * (N - 1) + j + i;

            clothMesh.vertices[index * 9] = cloth->vertices[vertex + N].x;
            clothMesh.vertices[index * 9 + 1] = cloth->vertices[vertex + N].y;
            clothMesh.vertices[index * 9 + 2] = cloth->vertices[vertex + N].z;
            clothMesh.normals[index * 9] = 0;
            clothMesh.normals[index * 9 + 1] = 0;
            clothMesh.normals[index * 9 + 2] = 1;
            clothMesh.texcoords[index * 6] = 0;
            clothMesh.texcoords[index * 6 + 1] = 0;

            clothMesh.vertices[index * 9 + 3] = cloth->vertices[vertex + N + 1].x;
            clothMesh.vertices[index * 9 + 4] = cloth->vertices[vertex + N + 1].y;
            clothMesh.vertices[index * 9 + 5] = cloth->vertices[vertex + N + 1].z;
            clothMesh.normals[index * 9 + 3] = 0;
            clothMesh.normals[index * 9 + 4] = 0;
            clothMesh.normals[index * 9 + 5] = 1;
            clothMesh.texcoords[index * 6 + 2] = 0;
            clothMesh.texcoords[index * 6 + 3] = 0;

            clothMesh.vertices[index * 9 + 6] = cloth->vertices[vertex + 1].x;
            clothMesh.vertices[index * 9 + 7] = cloth->vertices[vertex + 1].y;
            clothMesh.vertices[index * 9 + 8] = cloth->vertices[vertex + 1].z;
            clothMesh.normals[index * 9 + 6] = 0;
            clothMesh.normals[index * 9 + 7] = 0;
            clothMesh.normals[index * 9 + 8] = 1;
            clothMesh.texcoords[index * 6 + 4] = 0;
            clothMesh.texcoords[index * 6 + 5] = 0;
        }
    }

    UploadMesh(&clothMesh, true);

    return clothMesh;
}

static void UpdateClothMesh(Mesh* mesh, Cloth* cloth)
{
    for (int i = 0; i < N - 1; i++)
    {
        for (int j = 0; j < N - 1; j++)
        {
            int index1 = i * (N - 1) + j;
            int index2 = i * (N - 1) + j + (N - 1) * (N - 1);
            int vertex = i * (N - 1) + j + i;

            mesh->vertices[index1 * 9] = cloth->vertices[vertex].x;
            mesh->vertices[index1 * 9 + 1] = cloth->vertices[vertex].y;
            mesh->vertices[index1 * 9 + 2] = cloth->vertices[vertex].z;

            mesh->vertices[index1 * 9 + 3] = cloth->vertices[vertex + N].x;
            mesh->vertices[index1 * 9 + 4] = cloth->vertices[vertex + N].y;
            mesh->vertices[index1 * 9 + 5] = cloth->vertices[vertex + N].z;

            mesh->vertices[index1 * 9 + 6] = cloth->vertices[vertex + 1].x;
            mesh->vertices[index1 * 9 + 7] = cloth->vertices[vertex + 1].y;
            mesh->vertices[index1 * 9 + 8] = cloth->vertices[vertex + 1].z;

            mesh->vertices[index2 * 9] = cloth->vertices[vertex + N].x;
            mesh->vertices[index2 * 9 + 1] = cloth->vertices[vertex + N].y;
            mesh->vertices[index2 * 9 + 2] = cloth->vertices[vertex + N].z;

            mesh->vertices[index2 * 9 + 3] = cloth->vertices[vertex + N + 1].x;
            mesh->vertices[index2 * 9 + 4] = cloth->vertices[vertex + N + 1].y;
            mesh->vertices[index2 * 9 + 5] = cloth->vertices[vertex + N + 1].z;

            mesh->vertices[index2 * 9 + 6] = cloth->vertices[vertex + 1].x;
            mesh->vertices[index2 * 9 + 7] = cloth->vertices[vertex + 1].y;
            mesh->vertices[index2 * 9 + 8] = cloth->vertices[vertex + 1].z;
        }
    }

    UpdateMeshBuffer(*mesh, 0, mesh->vertices, sizeof(float) * mesh->vertexCount * 3, 0);
}

static void UpdateCloth(Cloth* cloth, Camera* camera)
{
    // Gravity
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int index = i * N + j;
            
            if (index >= N)
            {
                cloth->vertices[index].y -= gravity;
            }
        }
    }

    // Link Constraint
    for (int k = 0; k < iterations; k++)
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                int index = i * N + j;

                if (j + 1 < N)
                {
                    LinkConstraint(cloth, index, index + 1);
                }

                if (i + 1 < N)
                {
                    LinkConstraint(cloth, index, index + N);
                }

                if (index < N)
                {
                    cloth->vertices[index] = cloth->prevVertices[index];
                }
            }
        }
    }

    // Velocity
    Vector3 w0 = (Vector3){ -size, -size, 0.0f };
    Vector3 w1 = (Vector3){ -size, size, 0.0f };
    Vector3 w2 = (Vector3){ size, size, 0.0f };
    Vector3 w3 = (Vector3){ size, -size, 0.0f };

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int index = i * N + j;

            Ray ray = GetMouseRay(GetMousePosition(), *camera);

            RayCollision wallHitInfo = GetRayCollisionQuad(ray, w0, w1, w2, w3);

            Vector3 mouse = wallHitInfo.point;

            Vector3 diff = Vector3Subtract(cloth->vertices[index], mouse);

            if (Vector3Length(diff) < 0.5f)
            {
                mouse = (Vector3)
                {
                    -GetMouseDelta().x,
                    GetMouseDelta().y,
                    0
                };

                mouse = Vector3Scale(mouse, 0.01f);
            }
            else
            {
                mouse = Vector3Zero();
            }

            Vector3 temp;

            temp = cloth->vertices[index];

            cloth->vertices[index] = Vector3Add(cloth->vertices[index], Vector3Subtract(cloth->vertices[index], Vector3Add(cloth->prevVertices[index], mouse)));

            cloth->prevVertices[index] = temp;
        }
    }
}

static void LinkConstraint(Cloth* cloth, int p1, int p2)
{
    // Calculate the distance
    Vector3 diff = Vector3Subtract(cloth->vertices[p1], cloth->vertices[p2]);
    float d = Vector3Length(diff);

    // Difference scalar
    float difference = (spacing - d) / d;

    // Translation for each PointMass. They'll be pushed 1/2 the required distance to match their resting distances.
    Vector3 translate = Vector3Scale(diff, 0.5f * difference);

    cloth->vertices[p1] = Vector3Add(cloth->vertices[p1], translate);
    cloth->vertices[p2] = Vector3Subtract(cloth->vertices[p2], translate);
}