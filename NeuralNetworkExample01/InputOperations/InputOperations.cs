using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkExample01.InputOperations
{
    public class InputOperations
    {
        public string DefaultPath { get; set; }
        public InputOperations(string defaultPath = "features_read.csv") 
        {
            DefaultPath = defaultPath;
        }

        public void WriteFile(List<TestInput> inputes, string? path = null)
        {
            var file = new System.IO.StreamWriter(path ?? DefaultPath);
            foreach (var input in inputes)
            {
                file.WriteLine(input.ToCsvString());
            }
            file.Close();
        }

        public List<TestInput> ReadFile()
        {
            var testInputes = new List<TestInput>();
            const Int32 BufferSize = 128;
            using (var fileStream = File.OpenRead(DefaultPath))
            using (var streamReader = new StreamReader(fileStream, Encoding.UTF8, true, BufferSize))
            {
                String line;
                while ((line = streamReader.ReadLine()) != null)
                {
                    // Process line
                    var input = TestInput.ParseInput(line, 3, 2);
                    testInputes.Add(input);
                }
            }
            return testInputes;
        }
    }
}
