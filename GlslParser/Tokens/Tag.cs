namespace GlslParser.Tokens
{
    public enum Tag
    {
        // keywords
        Version,
        Flat,
        In,
        Out,
        Uniform,
        Struct,
        Define,

        // interpunction
        OpenBrace, CloseBrace,
        OpenBracket, CloseBracket,
        OpenParenthesis, CloseParenthesis,
        Hash,
        Semicolon,
        LineComment,
        End,
    }
}