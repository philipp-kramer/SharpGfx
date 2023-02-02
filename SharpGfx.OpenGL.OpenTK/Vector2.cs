using SharpGfx.Primitives;

namespace SharpGfx.OpenGL.OpenTK;

internal readonly struct Vector2 : IVector2
{
    public readonly global::OpenTK.Mathematics.Vector2 Value;
    private readonly Space _space;

    public Vector2(Space space, global::OpenTK.Mathematics.Vector2 value)
    {
        _space = space;
        Value = value;
    }

    Space IPrimitive.Space => _space;
    public float X => Value.X;
    public float Y => Value.Y;
    public float Length => Value.Length;

    IVector2 IVector2.Add(IVector2 r)
    {
        var ovr = (Vector2)r;
        return new Vector2(ovr._space, Value + ((Vector2)r).Value);
    }

    IVector2 IVector2.Sub(IVector2 r)
    {
        var ovr = (Vector2)r;
        return new Vector2(ovr._space, Value - ((Vector2)r).Value);
    }

    IVector2 IVector2.Mul(float scalar)
    {
        return new Vector2(_space, Value * scalar);
    }

    IVector2 IVector2.Mul(IVector2 r)
    {
        var ovr = (Vector2)r;
        return new Vector2(ovr._space, Value * ((Vector2)r).Value);
    }

    float IVector2.Dot(IVector2 r)
    {
        return global::OpenTK.Mathematics.Vector2.Dot(Value, ((Vector2)r).Value);
    }

    public IVector2 Normalized()
    {
        return new Vector2(_space, Value.Normalized());
    }
}