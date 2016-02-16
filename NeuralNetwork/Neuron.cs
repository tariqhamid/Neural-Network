namespace NeuralNetwork
{
    public struct Neuron
    {
        public float charge;
        public float bias;
        public ActivationType activation;

        public Neuron(ActivationType activation, float charge, float bias)
        {
            this.activation = activation;
            this.charge = charge;
            this.bias = bias;
        }
    }
}