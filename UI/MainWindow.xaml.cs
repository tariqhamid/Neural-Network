using System;
using System.Windows;
using System.Windows.Controls;
using System.IO;
using NN;
using System.Threading;
using System.Windows.Threading;
using System.Runtime.Serialization.Formatters.Binary;

namespace UI
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private NeuralNetwork neuralNetwork;
        private bool initilized = false;
        private double value1, value2, value3;
        private int value4;
        private double[][] data;

        public MainWindow()
        {
            InitializeComponent();
            build_button_Click(null, null);
            initilized = true;
        }

        private void train_button_Click(object sender, RoutedEventArgs e)
        {
            //Get File
            Microsoft.Win32.OpenFileDialog dlg = new Microsoft.Win32.OpenFileDialog();

            // Set filter for file extension and default file extension 
            dlg.DefaultExt = ".txt";
            dlg.Filter = "TXT Files (*.txt)|*.txt";

            // Display OpenFileDialog by calling ShowDialog method 
            Nullable<bool> result = dlg.ShowDialog();


            // Get the selected file name and display in a TextBox 
            if (result == true)
            {
                // Open document 
                string filename = dlg.FileName;

                //Train
                data = GetData(filename);
                MessageBox.Show("Trainning is about to start, note some values might be rounded to int e.g. particle amount.\nTrainning neural network can take several minutes depending on complexity.", "Important Message", MessageBoxButton.OK, MessageBoxImage.Information);
                neuralNetwork.progressChanged += NeuralNetwork_progressChanged;
                Thread t = new Thread(new ParameterizedThreadStart(TrainInThread));
                t.Start(comboBox.SelectedIndex);
            }
        }

        private void NeuralNetwork_progressChanged(double percent)
        {
            this.Dispatcher.Invoke(DispatcherPriority.Normal, new Action(delegate ()
            {
                this.progressBar.Value = percent;
                if (percent == 100)
                {
                    MessageBox.Show("Trainning Complete", "Info", MessageBoxButton.OK, MessageBoxImage.Information);
                }
            }));
        }

        private void TrainInThread(object obj)
        {
            switch ((int)obj)
            {
                case 0:
                    ((BackPropagation)neuralNetwork).Train(data, value1, value2, value3, value4);
                    break;
                default:
                case 1:
                    ((ParticleSwarmOptimisation)neuralNetwork).Train(data, (int)Math.Round(value1), value2, value3, value4);
                    break;
                case 2:
                    ((GeneticAlgorithm)neuralNetwork).Train(data, (int)Math.Round(value1), (int)Math.Round(value2), (int)Math.Round(value3));
                    break;
            }
        }

        private void compute_button_Click(object sender, RoutedEventArgs e)
        {
            Microsoft.Win32.OpenFileDialog dlg = new Microsoft.Win32.OpenFileDialog();

            // Set filter for file extension and default file extension 
            dlg.DefaultExt = ".txt";
            dlg.Filter = "TXT Files (*.txt)|*.txt";

            // Display OpenFileDialog by calling ShowDialog method 
            Nullable<bool> result = dlg.ShowDialog();


            // Get the selected file name and display in a TextBox 
            if (result == true)
            {
                // Open document 
                string filename = dlg.FileName;

                double[] results = neuralNetwork.ComputeResults(GetData(filename)[0]);

                TextWriter writer = new StreamWriter("results.txt");
                for (int i = 0; i < results.Length; i++)
                {
                    writer.Write(results[i] + ", ");
                }
                writer.Close();
                MessageBox.Show("Neural Network Successfully computed the results and put them into 'results.txt' relative to exe", "Operation Completed", MessageBoxButton.OK, MessageBoxImage.Information);
            }
        }

        private void build_button_Click(object sender, RoutedEventArgs e)
        {
            //Setup Nerual Network
            int input, hidden, output;

            if (!int.TryParse(input_textbox.Text, out input))
            {
                MessageBox.Show("Error: input is not in correct format.", "Input Value Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }
            if (!int.TryParse(hidden_textbox.Text, out hidden))
            {
                MessageBox.Show("Error: hidden is not in correct format.", "Input Value Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }
            if (!int.TryParse(output_textbox.Text, out output))
            {
                MessageBox.Show("Error: output is not in correct format.", "Input Value Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            switch (comboBox.SelectedIndex)
            {
                case 0:
                    neuralNetwork = new BackPropagation(input, hidden, output, (ActivationType)inputHiddenActivation.SelectedIndex, (ActivationType)hiddenOutputActivation.SelectedIndex);
                    break;
                default:
                case 1:
                    neuralNetwork = new ParticleSwarmOptimisation(input, hidden, output, (ActivationType)inputHiddenActivation.SelectedIndex, (ActivationType)hiddenOutputActivation.SelectedIndex);
                    break;
                case 2:
                    neuralNetwork = new GeneticAlgorithm(input, hidden, output, (ActivationType)inputHiddenActivation.SelectedIndex, (ActivationType)hiddenOutputActivation.SelectedIndex);
                    break;
            }

            //Setup Trainning Data
            if (!double.TryParse(value1_textbox.Text, out value1))
            {
                MessageBox.Show("Error: " + value1_label.Content + " is not in correct format.", "Input Value Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }
            if (!double.TryParse(value2_textbox.Text, out value2))
            {
                MessageBox.Show("Error: " + value2_label.Content + " is not in correct format.", "Input Value Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }
            if (!double.TryParse(value3_textbox.Text, out value3))
            {
                MessageBox.Show("Error: " + value3_label.Content + " is not in correct format.", "Input Value Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }
            if (!int.TryParse(value4_textbox.Text, out value4))
            {
                MessageBox.Show("Error: " + value4_label.Content + " is not in correct format.", "Input Value Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            train_button.IsEnabled = true;
            compute_button.IsEnabled = true;
            accuracy_button.IsEnabled = true;
        }

        private void save_button_Click(object sender, RoutedEventArgs e)
        {
            Microsoft.Win32.SaveFileDialog dlg = new Microsoft.Win32.SaveFileDialog();

            // Set filter for file extension and default file extension 
            dlg.DefaultExt = ".txt";
            dlg.Filter = "TXT Files (*.txt)|*.txt";

            // Display OpenFileDialog by calling ShowDialog method 
            Nullable<bool> result = dlg.ShowDialog();


            // Get the selected file name and display in a TextBox 
            if (result == true)
            {
                // Open document 
                string filename = dlg.FileName;
                byte[] data = ObjectToByteArray(neuralNetwork.GetWeights());
                File.WriteAllBytes(filename, data);
            }
        }

        private void load_button_Click(object sender, RoutedEventArgs e)
        {
            Microsoft.Win32.OpenFileDialog dlg = new Microsoft.Win32.OpenFileDialog();

            // Set filter for file extension and default file extension 
            dlg.DefaultExt = ".txt";
            dlg.Filter = "TXT Files (*.txt)|*.txt";

            // Display OpenFileDialog by calling ShowDialog method 
            Nullable<bool> result = dlg.ShowDialog();


            // Get the selected file name and display in a TextBox 
            if (result == true)
            {
                // Open document 
                string filename = dlg.FileName;
                byte[] data = File.ReadAllBytes(filename);
                neuralNetwork.SetWeights((double[])ByteArrayToObject(data));
            }
        }

        private double[][] GetData(string filename)
        {
            if (!File.Exists(filename))
            {
                throw new Exception("Error: file not found '" + filename + "'");
            }
            TextReader read = new StreamReader(filename);
            string s = read.ReadToEnd();
            read.Close();

            string[] data = s.Split(';');
            double[][] result = new double[data.Length][];

            for (int i = 0; i < data.Length; i++)
            {
                string[] io = data[i].Split(',');
                result[i] = new double[io.Length];
                for (int j = 0; j < io.Length; j++)
                {
                    result[i][j] = double.Parse(io[j]);
                }
            }
            return result;
        }

        private void comboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (initilized)
            {
                train_button.IsEnabled = false;
                compute_button.IsEnabled = false;
                accuracy_button.IsEnabled = false;
                switch (comboBox.SelectedIndex)
                {
                    case 0:
                        value1_label.Content = "Learning Rate";
                        value2_label.Content = "Momentum";
                        value3_label.Content = "Decay";
                        value4_label.Content = "Repeat";

                        value1_textbox.Text = "0.05";
                        value2_textbox.Text = "0.01";
                        value3_textbox.Text = "0.0001";
                        value4_textbox.Text = "5000";

                        value4_label.Visibility = Visibility.Visible;
                        value4_textbox.Visibility = Visibility.Visible;
                        break;
                    case 1:
                        value1_label.Content = "Amount Of Particles";
                        value2_label.Content = "Exit At Error";
                        value3_label.Content = "Death Probability";
                        value4_label.Content = "Repeat";

                        value1_textbox.Text = "12";
                        value2_textbox.Text = "0.01";
                        value3_textbox.Text = "0.005";
                        value4_textbox.Text = "1000";

                        value4_label.Visibility = Visibility.Visible;
                        value4_textbox.Visibility = Visibility.Visible;
                        break;
                    case 2:
                        value1_label.Content = "Generation Amount";
                        value2_label.Content = "Population Size";
                        value3_label.Content = "Mutation";
                        value4_label.Visibility = Visibility.Hidden;

                        value1_textbox.Text = "5000";
                        value2_textbox.Text = "20";
                        value3_textbox.Text = "5";
                        value4_textbox.Text = "0";
                        value4_textbox.Visibility = Visibility.Hidden;
                        break;
                }
            }
        }

        private void accuracy_button_Click(object sender, RoutedEventArgs e)
        {
            Microsoft.Win32.OpenFileDialog dlg = new Microsoft.Win32.OpenFileDialog();

            // Set filter for file extension and default file extension 
            dlg.DefaultExt = ".txt";
            dlg.Filter = "TXT Files (*.txt)|*.txt";

            // Display OpenFileDialog by calling ShowDialog method 
            Nullable<bool> result = dlg.ShowDialog();


            // Get the selected file name and display in a TextBox 
            if (result == true)
            {
                // Open document 
                string filename = dlg.FileName;

                double[][] data = GetData(filename);
                double accuracy = Math.Round(neuralNetwork.Accuracy(data) * 100, 2);
                MessageBox.Show("Neural Network's current accuracy while predicting data was " + accuracy, "Accuracy", MessageBoxButton.OK, MessageBoxImage.Information);
            }
        }

        private void inputHiddenActivation_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (initilized)
            {
                train_button.IsEnabled = false;
                compute_button.IsEnabled = false;
                accuracy_button.IsEnabled = false;
            }
        }

        private void hiddenOutputActivation_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (initilized)
            {
                train_button.IsEnabled = false;
                compute_button.IsEnabled = false;
                accuracy_button.IsEnabled = false;
            }
        }

        private void input_textbox_TextChanged(object sender, TextChangedEventArgs e)
        {
            if (initilized)
            {
                train_button.IsEnabled = false;
                compute_button.IsEnabled = false;
                accuracy_button.IsEnabled = false;
            }
        }

        private void hidden_textbox_TextChanged(object sender, TextChangedEventArgs e)
        {
            if (initilized)
            {
                train_button.IsEnabled = false;
                compute_button.IsEnabled = false;
                accuracy_button.IsEnabled = false;
            }
        }

        private void output_textbox_TextChanged(object sender, TextChangedEventArgs e)
        {
            if (initilized)
            {
                train_button.IsEnabled = false;
                compute_button.IsEnabled = false;
                accuracy_button.IsEnabled = false;
            }
        }

        // Convert an object to a byte array
        public static byte[] ObjectToByteArray(Object obj)
        {
            BinaryFormatter bf = new BinaryFormatter();
            using (var ms = new MemoryStream())
            {
                bf.Serialize(ms, obj);
                return ms.ToArray();
            }
        }

        // Convert a byte array to an Object
        public static Object ByteArrayToObject(byte[] arrBytes)
        {
            using (var memStream = new MemoryStream())
            {
                var binForm = new BinaryFormatter();
                memStream.Write(arrBytes, 0, arrBytes.Length);
                memStream.Seek(0, SeekOrigin.Begin);
                var obj = binForm.Deserialize(memStream);
                return obj;
            }
        }
    }
}
