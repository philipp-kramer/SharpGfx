namespace GlslParser.Tokens
{
    public enum Tag
    {
        // keywords
        Version,
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