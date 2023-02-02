namespace GlslParser.Tokens;

public class UnexpectedToken : Token
{
    public UnexpectedToken(Location location)
        : base(location)
    {
    }

    public override string ToString()
    {
        return "<unexpected token>";
    }
}