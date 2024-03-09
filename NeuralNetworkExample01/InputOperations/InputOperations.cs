using System.Text;

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


        public void WriteGraphsToFiles(Dictionary<string, List<string>> graphs)
        {
            foreach (var graph in graphs)
            {
                var file = new System.IO.StreamWriter(graph.Key + ".txt");
                for (var i = 0; i < graph.Value.Count; ++i)
                {
                    file.WriteLine(graph.Value[i]);
                }
                
                file.Close();
            }
            
        }

        //graphs

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

        public List<TestInput> ParseInputs(List<string> lines, int features, int outputs)
        {
            var result = new List<TestInput>();
            foreach (var line in lines)
            {
                // Process line
                var input = TestInput.ParseInput(line, features, outputs);
                result.Add(input);
            }
            return result;
        }

        public void WriteResultToDictionary(Dictionary<string, List<string>> map, string key, string value)
        {
            if (map.ContainsKey(key))
            {
                map[key].Add(value);
            }
            else
            {
                map[key] = new List<string> { value };
            }
        }
    }
}
