using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    public class RNN
    {
        public Network[] networks;

        public RNN(int input, int hidden, int output, int amount, bool connectAllNodes)
        {
            if (amount < 2)
                throw new Exception("Amount needs to be greater then 1");

            networks = new Network[amount + 1];
            networks[0] = new Network(new int[] { input, hidden }, connectAllNodes);
            for (int i = 1; i < networks.Length - 1; i++)
            {
                networks[i] = new Network(new int[] { input + hidden, hidden }, connectAllNodes);
            }            
            networks[networks.Length - 1] = new Network(new int[] { hidden, output }, connectAllNodes);
        }

        public float[] Compute(float[][] xValues)
        {
            if (xValues.Length != networks.Length - 1)
                throw new Exception("input's length length doesn't match RNN");

            float[][] outputs = new float[xValues.Length][];
            outputs[0] = networks[0].Compute(xValues[0]);

            for (int i = 1; i < xValues.Length - 1; i++)
                outputs[i] = networks[i].Compute(CombineLayers(xValues[i], outputs[i - 1]));

            return networks[networks.Length - 1].Compute(outputs[outputs.Length - 1]);
        }


        private float[] CombineLayers(float[] input, float[] hidden)
        {
            float[] rtn = new float[input.Length + hidden.Length];
            for (int i = 0; i < input.Length; i++)
            {
                rtn[i] = input[i];
            }
            for (int i = 0; i < hidden.Length; i++)
            {
                rtn[i + input.Length] = hidden[i];
            }

            return rtn;
        }
    }
}