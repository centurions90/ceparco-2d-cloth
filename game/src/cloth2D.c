#include <stdlib.h>
#include "raylib.h"
#include "raymath.h"

//----------------------------------------------------------------------------------
// Variables Definition
//----------------------------------------------------------------------------------
static const int screenWidth = 1920;
static const int screenHeight = 1080;
static int N;                           // Cloth resolution
static float size = 800;
static float* x;
static float* y;
static int* pinned;
static float spacing;
static int iterations = 1;

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
        N = atof(argv[1]);
    } else {
        N = 10;
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

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int index = i * N + j;
            DrawCircle(x[index], y[index], 3, RED);

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
    float originX = screenWidth / 2.0 - size / 2.0;
    float originY = 150;
    spacing = 1.0 / N * size;

    x = malloc(N * N * sizeof(float));
    y = malloc(N * N * sizeof(float));
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
        }
    }
}

static void FreeCloth(int N)
{
    free(x);
    free(y);
    free(pinned);
}

static void UpdateCloth()
{
    
}