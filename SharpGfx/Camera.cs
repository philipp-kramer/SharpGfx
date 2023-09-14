using System;
using SharpGfx.Primitives;

namespace SharpGfx;

public abstract class Camera
{
    private readonly float _halfFrustumHeight = MathF.Tan(MathF.PI / 8);

    public Point3 Position { get; set; }
    public IVector3 LookAt { get; set; }
    public Projection Projection { get; }

    protected Camera(IVector3 lookAt, Projection? projection = default)
    {
        LookAt = lookAt;
        Projection = projection ?? new PerspectiveProjection(MathF.PI / 4);
    }

    public (Point3, Point3, Point3, Point3) GetFrustum(IVector3 unitY, float aspect)
    {
        var up = (unitY - unitY.Dot(LookAt) * LookAt).Normalized();
        var right = LookAt.Cross(up);
        var center = Position + LookAt;
        float halfWidth = aspect * _halfFrustumHeight;

        var tl = center + _halfFrustumHeight * up - halfWidth * right;
        var tr = center + _halfFrustumHeight * up + halfWidth * right;
        var bl = center - _halfFrustumHeight * up - halfWidth * right;
        var br = center - _halfFrustumHeight * up + halfWidth * right;

        return (tl, tr, bl, br);
    }
}