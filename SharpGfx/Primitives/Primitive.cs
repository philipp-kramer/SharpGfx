namespace SharpGfx.Primitives
{
    public interface Primitive
    {
        protected internal Space Space { get; }
     
        public bool In(Space space)
        {
            return Space == space;
        }

        public static bool IsVisible(Primitive l, Primitive r)
        {
            return l.Space == r.Space || l.Space.Domain < r.Space.Domain && l.Space.Domain != Domain.Color;
        }
    }
}
