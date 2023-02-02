namespace GlslParser.Tree;

public class ArrayVariableNode : VariableNode
{
    public ArrayVariableNode(Location location, VariableNode variable)
        : base(location, variable.Type, variable.Identifier)
    {
    }

    public override string ToString()
    {
        return $"{base.ToString()}[]";
    }
}