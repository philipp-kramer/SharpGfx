using SharpGfx.Primitives;

namespace SharpGfx;

public class OrbitCamera : InteractiveCamera
{
    private const float MoveSensitivity = 1f/250;

    private float _radius;

    public Point3 Center { get; private set; }

    public OrbitCamera(Space world, Point3 center, float radius)
        : base(world, center + radius * world.Vector3(0, 0, 1))
    {
        Center = center;
        _radius = radius;
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
                    Pitch -= MoveSensitivity * deltaY;
                    Yaw += MoveSensitivity * deltaX;
                    break;

                case MouseButton.Right:
                    var lookAtX = LookAt.Cross(World.Unit3Y);
                    var lookAtY = LookAt.Cross(lookAtX);
                    Center -= MoveSensitivity * (deltaX * lookAtX + deltaY * lookAtY);
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
            UpdatePosition();
        }
    }

    private void UpdatePosition()
    {
        Position = Center - _radius * LookAt;
    }
}