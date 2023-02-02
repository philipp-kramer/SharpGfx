using System;
using System.Collections.Generic;
using System.Linq;

namespace GlslParser.Tree;

public class ProgramNode : Node
{
    public List<DeclarationNode> Declarations { get; }

    public ProgramNode(Location location, List<DeclarationNode> declarations)
        : base(location)
    {
        Declarations = declarations;
    }

    public override string ToString()
    {
        return Declarations.Aggregate(string.Empty, (a, b) => a + Environment.NewLine + b);
    }
}