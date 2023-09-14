using System;
using SharpGfx.Primitives;

namespace SharpGfx;

public abstract class InteractiveCamera : Camera
{
    private float _pitch;
    private float _yaw;
    public MouseButton MouseButtons { get; private set; }

    protected Space World { get; }
    public virtual (float x, float y) MousePosition { get; set; }
    public virtual float MouseScrollX { get; set; }
    public virtual float MouseScrollY { get; set; }

    protected InteractiveCamera(Space world, Point3 position, Projection? projection = default) 
        : base(world.Unit3Z, projection)
    {
        Position = position;
        World = world;
        Yaw = -MathF.PI / 2;
    }

    public virtual void KeyDown(ConsoleKey key) {}

    public virtual bool MouseDown(MouseButton button)
    {
        bool isChange = !MouseButtons.HasFlag(button);
        MouseButtons |= button;
        return isChange;
    }

    public virtual bool MouseUp(MouseButton button)
    {
        bool isChange = MouseButtons.HasFlag(button);
        MouseButtons &= ~button;
        return isChange;
    }

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