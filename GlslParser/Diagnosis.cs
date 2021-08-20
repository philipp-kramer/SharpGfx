using System;
using System.Collections.Generic;
using System.Linq;

namespace GlslParser
{
    public class Diagnosis
    {
        private struct Error
        {
            internal string Source { get; }
            internal string Message { get; }

            public Error(string source, string message)
            {
                Source = source;
                Message = message;
            }

            public override string ToString()
            {
                return $"{Source}: {Message}";
            }
        }

        private readonly List<Error> _errors = new List<Error>();

        public string Source { get; set; }
        public bool HasErrors => _errors.Any();

        public void ReportError(string message)
        {
            _errors.Add(new Error(Source, message));
        }

        public override string ToString()
        {
            return $"{_errors.Aggregate(string.Empty, (a, b) => a + Environment.NewLine + b)}";
        }
    }
}