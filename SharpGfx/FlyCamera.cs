using System;
using SharpGfx.Primitives;

namespace SharpGfx;

public class FlyCamera : InteractiveCamera
{
    private const float FlySpeed = 1f / 10;
    private const float MoveSensitivity = 1/250f;

    public FlyCamera(Space world, Point3 position, Projection? projection = default)
        : base(world, position, projection)
    {
    }

    public override void KeyDown(ConsoleKey key)
    {
        switch (key)
        {
            case ConsoleKey.D5:
            case ConsoleKey.NumPad5:
            case ConsoleKey.PageUp:
            case ConsoleKey.E:
                Position += FlySpeed * LookAt;
                break;

            case ConsoleKey.D0:
            case ConsoleKey.NumPad0:
            case ConsoleKey.PageDown:
            case ConsoleKey.Q:
                Position -= FlySpeed * LookAt;
                break;

            case ConsoleKey.D4:
            case ConsoleKey.NumPad4:
            case ConsoleKey.LeftArrow:
            case ConsoleKey.A:
                Position -= GetHorizontalDelta();
                break;

            case ConsoleKey.D6:
            case ConsoleKey.NumPad6:
            case ConsoleKey.RightArrow:
            case ConsoleKey.D:
                Position += GetHorizontalDelta();
                break;

            case ConsoleKey.D8:
            case ConsoleKey.NumPad8:
            case ConsoleKey.UpArrow:
            case ConsoleKey.W:
                Position += FlySpeed * World.Unit3Y;
                break;

            case ConsoleKey.D2:
            case ConsoleKey.NumPad2:
            case ConsoleKey.DownArrow:
            case ConsoleKey.S:
                Position -= FlySpeed * World.Unit3Y;
                break;
        }
    }

    public override (float x, float y) MousePosition
    {
        set
        {
            float deltaX = float.IsNaN(MousePosition.x) ? 0 : value.x - MousePosition.x;
            float deltaY = float.IsNaN(MousePosition.y) ? 0 : value.y - MousePosition.y;
            base.MousePosition = value;

            switch (MouseButtons.TryGetSingle())
            {
                case MouseButton.Left:
                    Yaw += deltaX * MoveSensitivity;
                    Pitch = Limit(Pitch - deltaY * MoveSensitivity, MathF.PI / 2);
                    break;

                case MouseButton.Middle:
                    break;
            }
        }
    }

    public override float MouseScrollY
    {
        set => Position -= value * LookAt;
    }

    private IVector3 GetHorizontalDelta()
    {
        return FlySpeed * IVector3.Cross(LookAt, World.Unit3Y).Normalized();
    }
}