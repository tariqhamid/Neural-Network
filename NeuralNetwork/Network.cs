using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    public class Network
    {
        
        public Neuron[][] neurons;
        public Synapsis[][] synapsis;

        public Network(int[] layers)
        {
            neurons = new Neuron[layers.Length][];
                        
        }
    }
}