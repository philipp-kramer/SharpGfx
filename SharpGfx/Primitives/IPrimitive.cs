namespace SharpGfx.Primitives
{
    public interface IPrimitive
    {
        protected internal Space Space { get; }
     
        public bool In(Space space)
        {
            return Space == space;
        }

        public bool In(Domain domain)
        {
            return Space.Domain == domain;
        }

        public static bool IsVisible(IPrimitive l, IPrimitive r)
        {
            return l.Space == r.Space || l.Space.Domain < r.Space.Domain && l.Space.Domain != Domain.Color;
        }
    }
}
