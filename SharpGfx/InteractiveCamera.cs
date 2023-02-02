using System;
using SharpGfx.Primitives;

namespace SharpGfx;

public abstract class InteractiveCamera : Camera
{
    private float _pitch;
    private float _yaw;
    protected Space World { get; }

    protected InteractiveCamera(Space world, Point3 position) 
    {
        Position = position;
        FovY = MathF.PI / 4;
        World = world;
        Yaw = -MathF.PI / 2;
    }

    public abstract void OnKeyDown(ConsoleKey key);
    public abstract void MouseDown(MouseButton button, float x, float y);
    public abstract void MouseDragging(MouseButton button, float x, float y);

    public float Pitch
    {
        get => _pitch;
        protected set
        {
            _pitch = Limit(value, (0.5f - 1e-6f) * MathF.PI);
            UpdateLookAt();
        }
    }

    public float Yaw
    {
        get => _yaw;
        protected set
        {
            _yaw = value;
            UpdateLookAt();
        }
    }

    private void UpdateLookAt()
    {
        LookAt = World.Vector3(
                MathF.Cos(_pitch) * MathF.Cos(_yaw),
                MathF.Sin(_pitch),
                MathF.Cos(_pitch) * MathF.Sin(_yaw))
            .Normalized();
    }

    protected static float Limit(float value, float range)
    {
        return Math.Min(Math.Max(value, -range), range);
    }
}