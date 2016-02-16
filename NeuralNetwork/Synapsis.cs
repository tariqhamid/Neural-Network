namespace NeuralNetwork
{
    public struct Synapsis
    {
        public float weight;
        public int start;
        public int end;

        public Synapsis(int start, int end, float weight)
        {
            this.start = start;
            this.end = end;
            this.weight = weight;
        }
    }
}