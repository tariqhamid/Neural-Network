using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    public class RNN
    {
        public Network[] networks;

        public RNN(int input, int hidden, int output, bool connectAllNodes)
        {
            networks = new Network[4];
            networks[0] = new Network(new int[] { input, hidden }, connectAllNodes);
            networks[1] = new Network(new int[] { input + hidden, hidden }, connectAllNodes);
            networks[2] = new Network(new int[] { input + hidden, hidden }, connectAllNodes);
            networks[3] = new Network(new int[] { hidden, output }, connectAllNodes);
        }

        public float[] Compute(float[] xValues1, float[] xValues2, float[] xValues3)
        {
            float[] outputs1 = networks[0].Compute(xValues1);
            float[] outputs2 = networks[1].Compute(CombineLayers(xValues2, outputs1));
            float[] outputs3 = networks[2].Compute(CombineLayers(xValues3, outputs2));
            return networks[3].Compute(outputs3);
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