namespace GlslParser.Tree;

public class ChannelNode : DeclarationNode
{
    public Direction Direction { get; }
    public VariableNode Variable { get; }

    public ChannelNode(Location location, Direction direction, VariableNode variable) 
        : base(location)
    {
        Direction = direction;
        Variable = variable;
    }

    public override string Name => Variable.Name;

    public override string ToString()
    {
        return $"{Direction} {Variable}";
    }
}