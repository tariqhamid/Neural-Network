using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    public enum ActivationType
    {
        None,
        LogisticSigmoid,
        HyperbolicTangent,
        HeavisideStep,
        Softmax
    }

    public class ActivationMethods
    {
        public static float[] Activation(float[] xValues, ActivationType type)
        {
            float[] rtn = new float[xValues.Length];
            switch (type)
            {
                case ActivationType.LogisticSigmoid:
                    for (int i = 0; i < xValues.Length; i++)
                        rtn[i] = LogisticSigmoid(xValues[i]);

                    return rtn;
                case ActivationType.HyperbolicTangent:
                    for (int i = 0; i < xValues.Length; i++)
                        rtn[i] = HyperbolidTangent(xValues[i]);

                    return rtn;
                case ActivationType.HeavisideStep:
                    for (int i = 0; i < xValues.Length; i++)
                        rtn[i] = HeavisideStep(xValues[i]);

                    return rtn;
                case ActivationType.Softmax:
                    return Softmax(xValues);
                default:
                    return xValues;
            }
        }

        private static float LogisticSigmoid(float values)
        {
            return 1.0f / (1.0f + Convert.ToSingle(Math.Exp(-values)));
        }

        private static float HyperbolidTangent(float values)
        {
            return Convert.ToSingle(Math.Tanh(values));
        }

        private static float HeavisideStep(float values)
        {
            if (values < 0)
                return 0;
            else
                return 1;
        }

        private static float[] Softmax(float[] values)
        {
            // determine max output sum
            // does all output nodes at once so scale doesn't have to be re-computed each time
            float max = values[0];
            for (int i = 1; i < values.Length; ++i)
                if (values[i] > max)
                    max = values[i];

            // determine scaling factor -- sum of exp(each val - max)
            float scale = 0.0f;
            for (int i = 0; i < values.Length; ++i)
                scale += Convert.ToSingle(Math.Exp(values[i] - max));

            float[] result = new float[values.Length];
            for (int i = 0; i < values.Length; ++i)
                result[i] = Convert.ToSingle(Math.Exp(values[i] - max)) / scale;

            return result; // now scaled so that xi sum to 1.0
        }
    }
}