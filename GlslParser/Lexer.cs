using System.Collections.Generic;
using System.Globalization;
using System.IO;
using GlslParser.Tokens;

namespace GlslParser
{
    public sealed class Lexer
    {
        private readonly TextReader _reader;
        private readonly List<string> _errors;
        private int _position;
        private char _current;
        private bool _endOfText;
        private int _tokenStart;

        public Lexer(TextReader reader, List<string> errors)
        {
            _reader = reader;
            _errors = errors;
            _position = 0;
            ReadNext();
        }

        private static readonly Dictionary<string, Tag> Keywords =
            new Dictionary<string, Tag> {
                { "version", Tag.Version },
                { "in", Tag.In },
                { "out", Tag.Out },
                { "uniform", Tag.Uniform },
                { "struct", Tag.Struct },
                { "define", Tag.Define }
            };

        internal void SkipToNewLine()
        {
            while (!_endOfText && _current != '\n')
            {
                ReadNext();
            }
        }

        internal Token Next()
        {
            while (true)
            {
                SkipBlanks();
                _tokenStart = _position;
                if (_endOfText)
                {
                    return new FixToken(new Location(_tokenStart, _position), Tag.End);
                }
                if (IsDigit(_current))
                {
                    return ReadInteger();
                }
                if (IsLetter(_current))
                {
                    return ReadName();
                }
                switch (_current)
                {
                    case '(':
                        return ReadFixToken(Tag.OpenParenthesis);
                    case ')':
                        return ReadFixToken(Tag.CloseParenthesis);
                    case '[':
                        return ReadFixToken(Tag.OpenBracket);
                    case ']':
                        return ReadFixToken(Tag.CloseBracket);
                    case '{':
                        return ReadFixToken(Tag.OpenBrace);
                    case '}':
                        return ReadFixToken(Tag.CloseBrace);
                    case '#':
                        return ReadFixToken(Tag.Hash);
                    case ';':
                        return ReadFixToken(Tag.Semicolon);
                    case '/':
                        ReadNext();
                        return _current == '/'
                            ? (Token)ReadFixToken(Tag.LineComment)
                            : new UnexpectedToken(Location);
                    default:
                        ReadNext();
                        return new UnexpectedToken(Location);
                }
            }
        }

        private Token ReadName()
        {
            var name = _current.ToString(CultureInfo.InvariantCulture);
            ReadNext();
            while (!_endOfText && (IsLetter(_current) || IsDigit(_current)))
            {
                name += _current;
                ReadNext();
            }
            if (Keywords.ContainsKey(name))
            {
                return new FixToken(Location, Keywords[name]);
            }
            return new IdentifierToken(Location, name);
        }

        private Location Location
        {
            get
            {
                return new Location(_tokenStart, _position);
            }
        }

        private IntegerToken ReadInteger()
        {
            var overflow = false;
            int value = _current - '0';
            ReadNext();
            while (!_endOfText && IsDigit(_current))
            {
                int digit = _current - '0';
                if (value > int.MaxValue / 10 || value * 10 - 1 > int.MaxValue - digit)
                {
                    overflow = true;
                }
                value = value * 10 + digit;
                ReadNext();
            }
            if (overflow)
            {
                Error(_tokenStart, "Too large integer value");
                value = int.MaxValue;
            }
            return new IntegerToken(new Location(_tokenStart, _position), value, false);
        }

        private void ReadNext()
        {
            var b = _reader.Read();
            if (b < 0)
            {
                _current = '\0';
                _endOfText = true;
            }
            else
            {
                _current = (char)b;
                _position++;
            }
        }

        private FixToken ReadFixToken(Tag tag)
        {
            ReadNext();
            return new FixToken(new Location(_tokenStart, _position), tag);
        }

        private static bool IsDigit(char c)
        {
            return '0' <= c && c <= '9';
        }

        private static bool IsLetter(char c)
        {
            return 'A' <= c && c <= 'Z' || 'a' <= c && c <= 'z';
        }

        private void SkipBlanks()
        {
            while (!_endOfText && _current <= ' ')
            {
                ReadNext();
            }
        }

        private void Error(int position, string message)
        {
            _errors.Add($"{message} POSITION {position}");
        }
    }
}
