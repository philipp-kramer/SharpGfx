namespace GlslParser.Tree;

public class IdentifierNode : Node
{
    public string Name { get; }

    public IdentifierNode(Location location, string name) 
        : base(location)
    {
        Name = name;
    }

    public override string ToString()
    {
        return Name;
    }
}