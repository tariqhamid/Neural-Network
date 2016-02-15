using System;

namespace NeuralNetwork
{
    public struct Synapsis
    {
        public float weight;
        public int start;
        public int end;

        public Synapsis(int start, int end)
        {
            this.start = start;
            this.end = end;
            Random rnd = new Random();
            weight = Convert.ToSingle(rnd.NextDouble());
        }
    }
}