namespace GlslParser.Tree
{
    public abstract class DeclarationNode : Node
    {
        protected DeclarationNode(Location location) 
            : base(location)
        {
        }

        public abstract string Name { get; }
    }
}
