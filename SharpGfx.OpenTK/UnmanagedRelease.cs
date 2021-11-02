using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpGfx.OpenTK
{
    internal static class UnmanagedRelease
    {
        private static readonly Queue<Action> Queue = new Queue<Action>();

        internal static void Add(Action pending)
        {
            Queue.Enqueue(pending);
        }

        internal static void ExecutePending()
        {
            while (Queue.Any())
            {
                Queue.Dequeue()();
            }
        }
    }
}