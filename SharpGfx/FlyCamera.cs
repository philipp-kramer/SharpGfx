using System;
using SharpGfx.Primitives;

namespace SharpGfx;

public class FlyCamera : InteractiveCamera
{
    private const float FlySpeed = 1f / 10;
    private const float MoveSensitivity = 1/250f;

    private float _mouseX = float.NaN;
    private float _mouseY = float.NaN;

    public FlyCamera(Space world, Point3 position)
        : base(world, position)
    {
    }

    public override void OnKeyDown(ConsoleKey key)
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

    public override void MouseDown(MouseButton button, float x, float y)
    {
        _mouseX = x;
        _mouseY = y;
    }

    public override void MouseDragging(MouseButton button, float x, float y)
    {
        float deltaX = float.IsNaN(_mouseX) ? 0 : x - _mouseX;
        float deltaY = float.IsNaN(_mouseY) ? 0 : y - _mouseY;

        switch (button)
        {
            case MouseButton.Left:
                Yaw += deltaX * MoveSensitivity;
                Pitch = Limit(Pitch - deltaY * MoveSensitivity, MathF.PI / 2);
                break;

            case MouseButton.Middle:
                Position -= y * LookAt;
                break;
        }

        MouseDown(button, x, y);
    }

    private IVector3 GetHorizontalDelta()
    {
        return FlySpeed * IVector3.Cross(LookAt, World.Unit3Y).Normalized();
    }
}