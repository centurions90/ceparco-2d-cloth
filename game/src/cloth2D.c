#include <stdlib.h>
#include <stdbool.h>
#include "raylib.h"
#include "raymath.h"

//----------------------------------------------------------------------------------
// Shared Variables Definition (global)
//----------------------------------------------------------------------------------
typedef struct cloth {
    int N;
    Vector2** positions;
    Vector2** lastPositions;
    float k;    // Stiffness
    float c;    // Damper
    int* pinned;
    int pinnedCount;
} Cloth;

//----------------------------------------------------------------------------------
// Local Variables Definition (local to this module)
//----------------------------------------------------------------------------------
static const int screenWidth = 1280;
static const int screenHeight = 720;
static Cloth cloth;
static float spacing;

//----------------------------------------------------------------------------------
// Local Functions Declaration
//----------------------------------------------------------------------------------

static void UpdateDrawFrame(void);                      // Update and draw one frame
static void CreateCloth(int N, float k, float damper); // Create cloth with an NxN resolution
static void UpdateCloth();
static bool isPinned(int* list, int count, int index);

//----------------------------------------------------------------------------------
// Main entry point
//----------------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    // Initialization
    //---------------------------------------------------------
    InitWindow(screenWidth, screenHeight, "2D Cloth Simulation");
    SetTargetFPS(60);

    int N;
    float k;
    float c;

    if (argc >= 2) {
        N = atof(argv[1]);
    } else {
        N = 10;
    }

    if (argc >= 3) {
        k = atof(argv[2]);
    } else {
        k = 0.1;
    }

    if (argc >= 4) {
        c = atof(argv[3]);
    } else {
        c = 0.01;
    }

    CreateCloth(N, k, c);

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
    free(cloth.pinned);
    for (int i = 0; i < N; i++) {
        free(cloth.positions[i]);
        free(cloth.lastPositions[i]);
    }

    free(cloth.positions);
    free(cloth.lastPositions);

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

    for (int i = 0; i < cloth.N; i++) {
        for (int j = 0; j < cloth.N; j++) {
            Vector2 point = cloth.positions[i][j];

            DrawCircle(point.x, point.y, 3, RED);
            
            if (i + 1 < cloth.N) {
                Vector2 next = cloth.positions[i + 1][j];
                DrawLine(point.x, point.y, next.x, next.y, RED);
            }

            if (j + 1 < cloth.N) {
                Vector2 next = cloth.positions[i][j + 1];
                DrawLine(point.x, point.y, next.x, next.y, RED);
            }
        }
    }

    DrawFPS(10, 10);

    EndDrawing();
    //----------------------------------------------------------------------------------
}

static void CreateCloth(int N, float k, float c) {
    Vector2 center = {screenWidth / 2, screenHeight / 2};
    Vector2 offset = {200, 200};
    spacing = ((1.0 / (N - 1) - 0.5) * 2) * offset.x - ((0 - 0.5) * 2) * offset.x;

    cloth.N = N;
    cloth.k = k;
    cloth.c = c;
    cloth.pinned = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        cloth.pinned[i] = i;
    }
    cloth.pinnedCount = N;
    cloth.positions = (Vector2**)malloc(N * sizeof(Vector2*));

    for (int i = 0; i < N; i++) {
        cloth.positions[i] = (Vector2*)malloc(N * sizeof(Vector2));

        for (int j = 0; j < N; j++) {
            cloth.positions[i][j] = (Vector2){
                .x = center.x + (((float)j / (N - 1) - 0.5) * 2) * offset.x,
                .y = center.y + (((float)i / (N - 1) - 0.5) * 2) * offset.y
            };
        }
    }

    cloth.lastPositions = (Vector2**)malloc(N * sizeof(Vector2*));
    for (int i = 0; i < N; i++) {
        cloth.lastPositions[i] = (Vector2*)malloc(N * sizeof(Vector2));

        for (int j = 0; j < N; j++) {
            cloth.lastPositions[i][j] = (Vector2){
                .x = center.x + ((((float)j / (N - 1)) - 0.5) * 2) * offset.x,
                .y = center.y + ((((float)i / (N - 1)) - 0.5) * 2) * offset.y
            };
        }
    }
}

static void UpdateCloth() {
    for (int i = 0; i < cloth.N; i++) {
        for (int j = 0; j < cloth.N; j++) {
            if (!isPinned(cloth.pinned, cloth.pinnedCount, i * cloth.N + j)) {
                Vector2 force;
                Vector2 spring = {0, 0};

                // Velocity
                force = Vector2Subtract(cloth.positions[i][j], cloth.lastPositions[i][j]);

                // Gravity
                //force = Vector2Add(force, (Vector2) {.x = 0, .y = 9.81 * GetFrameTime()});

                // Cloth
                for (int y = -1; y <= 1; y++) {
                    for (int x = -1; x <= 1; x++) {
                        if (
                            j + x >= 0 &&
                            j + x < cloth.N &&
                            i + y >= 0 &&
                            i + y < cloth.N
                        ) {
                            // Diagonal Points
                            if (abs(x) + abs(y) == 2) {
                                
                            }

                            // Horizontal/Vertical Points
                            else if (abs(x) + abs(y) == 1) {
                                
                            }
                        }
                    }
                }

                force = Vector2Add(force, spring);
                //force = Vector2Scale(force, GetFrameTime() * GetFrameTime());
                cloth.positions[i][j] = Vector2Add(cloth.positions[i][j], force);
                cloth.lastPositions[i][j] = cloth.positions[i][j];
            }
        }
    }
}

static bool isPinned(int* list, int count, int index) {
    bool result = false;

    for (int i = 0; i < count; i++) {
        if (list[i] == index) {
            result = true;
            break;
        }
    }

    return result;
}
