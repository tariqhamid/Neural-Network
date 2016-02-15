using System;

namespace NeuralNetwork
{
    public struct Neuron
    {
        public float charge;
        public float bias;
        public ActivationType activation;

        public Neuron(ActivationType activation)
        {
            this.activation = activation;
            Random rnd = new Random();
            charge = Convert.ToSingle(rnd.NextDouble());
            bias = Convert.ToSingle(rnd.NextDouble());
        }
    }
}