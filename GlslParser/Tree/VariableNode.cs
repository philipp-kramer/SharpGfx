namespace GlslParser.Tree;

public class VariableNode : DeclarationNode
{
    public IdentifierNode Type { get; }
    public IdentifierNode Identifier { get; }

    public VariableNode(Location location, IdentifierNode type, IdentifierNode identifier) :
        base(location)
    {
        Type = type;
        Identifier = identifier;
    }

    public override string Name => Identifier.Name;

    public override string ToString()
    {
        return $"{Type} {Identifier};";
    }
}