using SharpGfx.Primitives;

namespace SharpGfx.OpenTK
{
    internal readonly struct OtkVector2 : Vector2
    {
        public readonly global::OpenTK.Vector2 Value;
        private readonly Space _space;

        public OtkVector2(Space space, global::OpenTK.Vector2 value)
        {
            _space = space;
            Value = value;
        }

        Space Primitive.Space => _space;
        public float X => Value.X;
        public float Y => Value.Y;
        public float Length => Value.Length;

        Vector2 Vector2.Add(Vector2 r)
        {
            var ovr = (OtkVector2)r;
            return new OtkVector2(ovr._space, Value + ((OtkVector2)r).Value);
        }

        Vector2 Vector2.Sub(Vector2 r)
        {
            var ovr = (OtkVector2)r;
            return new OtkVector2(ovr._space, Value - ((OtkVector2)r).Value);
        }

        Vector2 Vector2.Mul(float scalar)
        {
            return new OtkVector2(_space, Value * scalar);
        }

        Vector2 Vector2.Mul(Vector2 r)
        {
            var ovr = (OtkVector2)r;
            return new OtkVector2(ovr._space, Value * ((OtkVector2)r).Value);
        }

        float Vector2.Dot(Vector2 r)
        {
            return global::OpenTK.Vector2.Dot(Value, ((OtkVector2)r).Value);
        }

        public Vector2 Normalized()
        {
            return new OtkVector2(_space, Value.Normalized());
        }
    }
}