using SharpGfx.Primitives;

namespace SharpGfx;

public class OrbitCamera : InteractiveCamera
{

    private float _radius;

    public Point3 Center { get; private set; }

    public OrbitCamera(Space world, Point3 center, float radius, Projection? projection = default)
        : base(world, center + radius * world.Vector3(0, 0, 1), projection)
    {
        Center = center;
        _radius = radius;
    }

    public override (float x, float y) MousePosition
    {
        set
        {
            const float moveSensitivity = 1f / 250;

            float deltaX = float.IsNaN(MousePosition.x) ? 0 : value.x - MousePosition.x;
            float deltaY = float.IsNaN(MousePosition.y) ? 0 : value.y - MousePosition.y;
            base.MousePosition = value;

            switch (MouseButtons.TryGetSingle())
            {
                case MouseButton.Left:
                    Pitch -= moveSensitivity * deltaY;
                    Yaw += moveSensitivity * deltaX;
                    break;

                case MouseButton.Right:
                    var lookAtX = LookAt.Cross(World.Unit3Y);
                    var lookAtY = LookAt.Cross(lookAtX);
                    Center -= moveSensitivity * (deltaX * lookAtX + deltaY * lookAtY);
                    break;
            }

            UpdatePosition();
        }
    }

    public override float MouseScrollY
    {
        set
        {
            _radius += value;
            if (Projection is OrthographicProjection orthographic)
            {
                const float sensitivity = 1f/10f;
                orthographic.PixelScale *= 1 + sensitivity * value;
            }
            UpdatePosition();
        }
    }

    private void UpdatePosition()
    {
        Position = Center - _radius * LookAt;
    }
}