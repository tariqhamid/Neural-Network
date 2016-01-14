using System;
using System.Text;

namespace NN
{
    /// <summary>
    /// Use methods inside this class to prep data before feeding it to neural network
    /// Normalise, Encode, Decode
    /// </summary>
    public class Data
    {
        /// <summary>
        /// Normalises data between 0 and 1
        /// </summary>
        /// <param name="data">Data that needs to be normalised</param>
        /// <returns>data after normalise</returns>
        public static double[] Normalise(double[] data)
        {
            int max = 0;
            for (int i = 1; i < data.Length; i++)
            {
                if (data[max] < data[i])
                    max = i;
            }
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = data[i] / data[max];
            }

            return data;
        }

        /// <summary>
        /// Encodes string into a array of doubles
        /// </summary>
        /// <param name="data">string that needs to be encoded</param>
        /// <returns>array or doubles after encoding</returns>
        public static double[] Encode(string data)
        {
            //To Bytes
            byte[] bytes = Encoding.Unicode.GetBytes(data);

            //To Double
            double[] converted = new double[bytes.Length];

            for (int i = 0; i < converted.Length; i++)
            {
                converted[i] = Convert.ToDouble(bytes[i]);
            }

            return converted;
        }

        /// <summary>
        /// Decodes a encoded string from array of doubles into a string
        /// </summary>
        /// <param name="data">array of doubles needed to be decoded</param>
        /// <returns>decoded string</returns>
        public static string Decode(double[] data)
        {
            //To Bytes
            byte[] bytes = new byte[data.Length];

            for (int i = 0; i < bytes.Length; i++)
            {
                bytes[i] = Convert.ToByte(data[i]);
            }

            //To String
            string converted = Encoding.Unicode.GetString(bytes);

            return converted;
        }
    }
}