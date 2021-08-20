namespace GlslParser.Tree
{
    public abstract class Node
    {
        public Location Location { get; }

        protected Node(Location location)
        {
            Location = location;
        }

        public abstract override string ToString();
    }
}