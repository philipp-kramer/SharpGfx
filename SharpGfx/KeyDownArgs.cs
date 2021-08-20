using System;

namespace SharpGfx
{
    public class KeyDownArgs : EventArgs
    {
        public ConsoleKey Key { get; }
        public bool Handled { get; set; }

        public KeyDownArgs(ConsoleKey key)
        {
            Key = key;
        }
    }
}