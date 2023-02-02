namespace GlslParser.Tokens;

public class IdentifierToken : Token
{
    public string Name { get; }

    public IdentifierToken(Location location, string name) :
        base(location)
    {
        Name = name;
    }

    public override string ToString()
    {
        return $"IDENTIFIER {Name}";
    }
}