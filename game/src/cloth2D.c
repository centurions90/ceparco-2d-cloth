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
static double totalTime = 0;

//----------------------------------------------------------------------------------
// Functions Declaration
//----------------------------------------------------------------------------------

static void UpdateDrawFrame(void);  // Update and draw one frame
static void CreateCloth(int N);     // Create cloth with an NxN resolution
static void FreeCloth(int N);       // Frees cloth from memory
static void UpdateCloth();          // Update cloth physics
static void LinkConstraint(int p1, int p2);

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
        N = atof(argv[1]);
    }
    else
    {
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

    clock_t start, end;
    double timeTaken;
    int count = 0;

    // Main game loop
    while (!WindowShouldClose())    // Detect window close button or ESC key
    {
        start = clock();
        UpdateCloth();
        end = clock();
        timeTaken = (double)(end - start) * 1e6 / CLOCKS_PER_SEC;
        totalTime += timeTaken;
        count++;
        UpdateDrawFrame();
    }

    // De-Initialization
    //--------------------------------------------------------------------------------------
    FreeCloth(N);

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
static void UpdateDrawFrame(void)
{
    // Draw
    //----------------------------------------------------------------------------------
    BeginDrawing();

    ClearBackground(BLACK);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int index = i * N + j;
            //DrawCircle(x[index], y[index], 3.0f, RED);

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
    spacing = 1.0 / N * size;

    x = malloc(N * N * sizeof(float));
    y = malloc(N * N * sizeof(float));
    prevx = malloc(N * N * sizeof(float));
    prevy = malloc(N * N * sizeof(float));
    pinned = malloc(N * N * sizeof(int));

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
    free(x);
    free(y);
    free(prevx);
    free(prevy);
    free(pinned);
}

static void UpdateCloth()
{
    // Gravity
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int index = i * N + j;

            if (!pinned[index])
            {
                y[index] = y[index] + gravity;
            }
        }
    }

    // Link Constraint
    //int max = (int)fmaxf(hIterations, vIterations);

    for (int k = 0; k < hIterations; k++)
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                int index = i * N + j;

                if (j + 1 < N)
                {
                    LinkConstraint(index, index + 1);
                }

                if (pinned[index])
                {
                    x[index] = prevx[index];
                    y[index] = prevy[index];
                }
            }
        }
    }

    for (int k = 0; k < vIterations; k++)
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                int index = i * N + j;

                if (i + 1 < N)
                {
                    LinkConstraint(index, index + N);
                }

                if (pinned[index])
                {
                    x[index] = prevx[index];
                    y[index] = prevy[index];
                }
            }
        }
    }

    // Velocity
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int index = i * N + j;
            float mouseX = 0.0f;
            float mouseY = 0.0f;

            float diffX = x[index] - GetMouseX();
            float diffY = y[index] - GetMouseY();
            if (diffX * diffX + diffY * diffY < 200.0f)
            {
                mouseX = GetMouseDelta().x;
                mouseY = GetMouseDelta().y;
            }

            float tempx;
            float tempy;

            tempx = x[index];
            tempy = y[index];

            x[index] += x[index] - prevx[index] + mouseX;
            y[index] += y[index] - prevy[index] + mouseY;

            prevx[index] = tempx;
            prevy[index] = tempy;
        }
    }
}

static void LinkConstraint(int p1, int p2)
{
    // calculate the distance
    float diffX = x[p1] - x[p2];
    float diffY = y[p1] - y[p2];
    float d = sqrtf(diffX * diffX + diffY * diffY);

    // difference scalar
    float difference = (spacing - d) / d;

    // translation for each PointMass. They'll be pushed 1/2 the required distance to match their resting distances.
    float translateX = diffX * 0.5 * difference;
    float translateY = diffY * 0.5 * difference;

    x[p1] += translateX;
    y[p1] += translateY;

    x[p2] -= translateX;
    y[p2] -= translateY;
}