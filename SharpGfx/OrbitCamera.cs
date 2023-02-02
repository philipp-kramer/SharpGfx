using System;
using SharpGfx.Primitives;

namespace SharpGfx;

public class OrbitCamera : InteractiveCamera
{
    private const float MoveSensitivity = 1f/250;

    private float _mouseX;
    private float _mouseY;
    private float _radius;

    public Point3 Center { get; private set; }

    public OrbitCamera(Space world, Point3 center, float radius)
        : base(world, center + radius * world.Vector3(0, 0, 1))
    {
        Center = center;
        _radius = radius;
    }

    public override void OnKeyDown(ConsoleKey key)
    {
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
                Pitch -= MoveSensitivity * deltaY;
                Yaw += MoveSensitivity * deltaX;
                break;

            case MouseButton.Middle:
                _radius += y;
                break;

            case MouseButton.Right:
                var lookAtX = LookAt.Cross(World.Unit3Y);
                var lookAtY = LookAt.Cross(lookAtX);
                Center -= MoveSensitivity * (deltaX * lookAtX + deltaY * lookAtY);
                break;
        }

        MouseDown(button, x, y);

        UpdatePosition();
    }

    private void UpdatePosition()
    {
        Position = Center - _radius * LookAt;
    }
}