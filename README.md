# SharpGfx
This is a little .NET / OpenGL graphics framework. The framework has some similarity to three.js. It is used for educational purposes. It will hopefully be developed further over the years.

There are currently two OpenGL backens: one targetting OpenGL directly and the other using OpenTK, a quite widely used OpenGL abstraction layer for .NET. The frawework can be used under Windows and Linux (the OpenTK backend obviously depends on the Linux support of OpenTK).

It is the intension to target at least one more low-level graphics API. Hence some efforts have already been made to leave this open, but the design is currently clearly geared towards OpenGL. Expect breaking changes in the future.

Check out the Wiki for an example use.
