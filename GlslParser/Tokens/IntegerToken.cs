namespace GlslParser.Tokens;

public class IntegerToken : Token
{
    public int Value { get; }
    public bool Hex { get; }

    public IntegerToken(Location location, int value, bool hex) :
        base(location)
    {
        Value = value;
        Hex = hex;
    }

    public override string ToString()
    {
        return $"INTEGER {Value}";
    }
}