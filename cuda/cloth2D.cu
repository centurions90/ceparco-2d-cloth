#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../raylib-master/src/raylib.h"

//----------------------------------------------------------------------------------
// Variables Definition
//----------------------------------------------------------------------------------
static const int screenWidth = 1920;
static const int screenHeight = 1080;
static int N;                           // Cloth resolution
static float size = 800.0f;
static float* x;
static float* y;
static float* prevx;
static float* prevy;
static int* pinned;
static float spacing;
static int hIterations = 30;
static int vIterations = 30;
static float gravity = 9.81f * 1.0f / 60.0f;
static int numThreads = 256;

//----------------------------------------------------------------------------------
// Functions Declaration
//----------------------------------------------------------------------------------

static void UpdateDrawFrame(void);  // Update and draw one frame
static void CreateCloth(int N);     // Create cloth with an NxN resolution
static void FreeCloth(int N);       // Frees cloth from memory
static void UpdateCloth();          // Update cloth physics

//----------------------------------------------------------------------------------
// Main entry point
//----------------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    // Initialization
    //---------------------------------------------------------
    InitWindow(screenWidth, screenHeight, "2D Cloth Simulation");
    SetTargetFPS(60);

    if (argc >= 2) {
        N = atoi(argv[1]);
    }
    else {
        N = 10;
    }

    if (argc >= 3)
    {
        hIterations = atoi(argv[2]);
    }

    if (argc >= 4)
    {
        vIterations = atoi(argv[3]);
    }

    CreateCloth(N);

    // Load global data (assets that must be available in all screens, i.e. font)
    //--------------------------------------------------------------------------------------

    // Main game loop
    while (!WindowShouldClose())    // Detect window close button or ESC key
    {
        UpdateCloth();
        UpdateDrawFrame();
    }

    // De-Initialization
    //--------------------------------------------------------------------------------------
    FreeCloth(N);

    // Unload global data loaded

    CloseWindow();          // Close window and OpenGL context
    //--------------------------------------------------------------------------------------

    return 0;
}

//----------------------------------------------------------------------------------
// Module specific Functions Definition
//----------------------------------------------------------------------------------
// Update and draw game frame
static void UpdateDrawFrame(void)
{
    // Draw
    //----------------------------------------------------------------------------------
    BeginDrawing();

    ClearBackground(BLACK);

    cudaDeviceSynchronize();

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int index = i * N + j;

            if (j + 1 < N)
            {
                int next = index + 1;
                DrawLine(x[index], y[index], x[next], y[next], RED);
            }

            if (i + 1 < N)
            {
                int next = index + N;
                DrawLine(x[index], y[index], x[next], y[next], RED);
            }
        }
    }

    DrawFPS(30, 30);

    EndDrawing();
    //----------------------------------------------------------------------------------
}

static void CreateCloth(int N)
{
    float originX = screenWidth / 2.0f - size / 2.0f;
    float originY = 150.0f;
    spacing = 1.0f / N * size;

    cudaMallocManaged(&x, N * N * sizeof(float));
    cudaMallocManaged(&y, N * N * sizeof(float));
    cudaMallocManaged(&prevx, N * N * sizeof(float));
    cudaMallocManaged(&prevy, N * N * sizeof(float));
    cudaMallocManaged(&pinned, N * N * sizeof(int));

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (i == 0)
            {
                pinned[j] = 1;
            }
            else
            {
                pinned[i * N + j] = 0;
            }

            int index = i * N + j;
            x[index] = originX + spacing * j;
            y[index] = originY + spacing * i;
            prevx[index] = x[index];
            prevy[index] = y[index];
        }
    }
}

static void FreeCloth(int N)
{
    cudaFree(x);
    cudaFree(y);
    cudaFree(prevx);
    cudaFree(prevy);
    cudaFree(pinned);
}

__global__ void SetGravity(int N, float* y, int* pinned, float gravity)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride)
    {
        if (!pinned[i])
        {
            y[i] = y[i] + gravity;
        }
    }
}

__global__ void SetLinkContraintH(int N, float* x, float* y, float* prevx, float* prevy, float spacing)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride)
    {
        int side = sqrtf(N);
        
        if (i % side != side - 1)
        {
            float diffX = x[i] - x[i + 1];
            float diffY = y[i] - y[i + 1];
            float d = sqrtf(diffX * diffX + diffY * diffY);
            float difference = (spacing - d) / d;
            float translateX = diffX * 0.5 * difference;
            float translateY = diffY * 0.5 * difference;
            x[i] += translateX;
            y[i] += translateY;
            x[i + 1] -= translateX;
            y[i + 1] -= translateY;
        }
    }
}

__global__ void SetLinkContraintV(int N, float* x, float* y, float* prevx, float* prevy, float spacing)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride)
    {
        int side = sqrtf(N);

        if (i < N - side)
        {
            float diffX = x[i] - x[i + side];
            float diffY = y[i] - y[i + side];
            float d = sqrtf(diffX * diffX + diffY * diffY);
            float difference = (spacing - d) / d;
            float translateX = diffX * 0.5 * difference;
            float translateY = diffY * 0.5 * difference;
            x[i] += translateX;
            y[i] += translateY;
            x[i + side] -= translateX;
            y[i + side] -= translateY;
        }
    }
}

__global__ void SetVelocity(int N, float* x, float* y, float* prevx, float* prevy, int mouseX, int mouseY, float mouseDeltaX, float mouseDeltaY)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride)
    {
        float mouseFinalX = 0.0f;
        float mouseFinalY = 0.0f;

        float diffX = x[i] - mouseX;
        float diffY = y[i] - mouseY;

        if (diffX * diffX + diffY * diffY < 200.0f)
        {
            mouseFinalX = mouseDeltaX;
            mouseFinalY = mouseDeltaY;
        }

        float tempx;
        float tempy;

        tempx = x[i];
        tempy = y[i];

        x[i] += x[i] - prevx[i] + mouseFinalX;
        y[i] += y[i] - prevy[i] + mouseFinalY;

        prevx[i] = tempx;
        prevy[i] = tempy;
    }
}

__global__ void ResetPinned(int N, float* x, float* y, float* prevx, float* prevy, int* pinned)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride)
    {
        if (pinned[i])
        {
            x[i] = prevx[i];
            y[i] = prevy[i];
        }
    }
}

static void UpdateCloth()
{   
    int numBlocks = (N * N + numThreads - 1) / numThreads;

    // Gravity
    SetGravity <<<numBlocks, numThreads>>> (N * N, y, pinned, gravity);
    //cudaDeviceSynchronize();

    // Link Constraint
    int max = fmaxf(hIterations, vIterations);
    int h = 0;
    int v = 0;

    for (int i = 0; i < max; i++)
    {
        if (h < hIterations)
        {
            SetLinkContraintH << <numBlocks, numThreads >> > (N * N, x, y, prevx, prevy, spacing);
            h++;
        }
        
        if (v < vIterations)
        {
            SetLinkContraintV << <numBlocks, numThreads >> > (N * N, x, y, prevx, prevy, spacing);
            v++;
        }

        //cudaDeviceSynchronize();
        ResetPinned <<<numBlocks, numThreads >> > (N, x, y, prevx, prevy, pinned);
    }

    // Velocity
    int mouseX = GetMouseX();
    int mouseY = GetMouseY();
    float mouseDeltaX = GetMouseDelta().x;
    float mouseDeltaY = GetMouseDelta().y;
    SetVelocity <<<numBlocks, numThreads>>> (N * N, x, y, prevx, prevy, mouseX, mouseY, mouseDeltaX, mouseDeltaY);
    //cudaDeviceSynchronize();
}