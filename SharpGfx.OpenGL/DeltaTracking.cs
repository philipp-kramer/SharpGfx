namespace SharpGfx.OpenGL
{
    internal class DeltaTracking
    {
        private int _x = int.MinValue;
        private int _y = int.MinValue;
        private int _deltaX;
        private int _deltaY;

        public int X
        {
            get => _x;
            set
            {
                if (_x != int.MinValue) _deltaX += _x - value;
                _x = value;
            }
        }

        public int Y
        {
            get => _y;
            set
            {
                if (_y != int.MinValue) _deltaY += _y - value;
                _y = value;
            }
        }

        public int DeltaX
        {
            get
            {
                int delta = _deltaX;
                _deltaX = 0;
                return delta;
            }
        }

        public int DeltaY
        {
            get
            {
                int delta = _deltaY;
                _deltaY = 0;
                return delta;
            }
        }
    }
}
