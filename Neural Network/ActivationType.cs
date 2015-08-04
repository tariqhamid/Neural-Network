using System;
using System.Collections.Generic;
using System.Text;

namespace NN
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
        public static double[] Softmax(double[] values)
        {
            // determine max output sum
            // does all output nodes at once so scale doesn't have to be re-computed each time
            double max = values[0];
            for (int i = 0; i < values.Length; ++i)
                if (values[i] > max) max = values[i];

            // determine scaling factor -- sum of exp(each val - max)
            double scale = 0.0;
            for (int i = 0; i < values.Length; ++i)
                scale += Math.Exp(values[i] - max);

            double[] result = new double[values.Length];
            for (int i = 0; i < values.Length; ++i)
                result[i] = Math.Exp(values[i] - max) / scale;

            return result; // now scaled so that xi sum to 1.0

        }

        public static double HyperbolicTangent(double value)
        {
            if (value < -20.0) return -1.0; // approximation is correct to 30 decimals
            else if (value > 20.0) return 1.0;
            else return Math.Tanh(value);
        }

        public static double LogisticSigmoid(double value)
        {
            return 1.0 / (1.0 + Math.Exp(-value));
        }

        public static double HeavisideStep(double values)
        {
            if (values < 0)
                return 0;
            else
                return 1;
        }

        ///===========
        ///Derivatives
        ///===========

        public static double LogisticSigmoidDerivative(double value)
        {
            return LogisticSigmoid(value) * (1 - LogisticSigmoid(value));
        }

        public static double HyperbolicTangentDerivative(double value)
        {
            return 1 - Math.Pow(HyperbolicTangent(value), 2);
        }
    }
}
