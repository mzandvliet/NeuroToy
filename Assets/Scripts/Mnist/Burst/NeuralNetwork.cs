

using System.Collections.Generic;
using Unity.Collections;

namespace NNBurst {
    public static class NeuralUtils {
        public static void Initialize(NativeNetwork net, System.Random random) {
            // Todo: init as jobs too. Needs Burst-compatible RNG.

            for (int l = 0; l < net.Layers.Length; l++) {
                NeuralMath.RandomGaussian(random, net.Layers[l].Weights, 0f, 1f);
                NeuralMath.RandomGaussian(random, net.Layers[l].Biases, 0f, 1f);
            }
        }
    }

    public struct NativeLayerConfig {
        public int Neurons;
    }

    public class NativeNetworkConfig {
        public List<NativeLayerConfig> Layers;

        public NativeNetworkConfig() {
            Layers = new List<NativeLayerConfig>();
        }
    }

    // Todo: could test layers just existing of slices of single giant arrays
    public class NativeNetworkLayer : System.IDisposable {
        public NativeArray<float> Biases;
        public NativeArray<float> Weights;
        public NativeArray<float> Outputs;
        public readonly int NumNeurons;
        public readonly int NumInputs;

        public NativeNetworkLayer(int numNeurons, int numInputs) {
            Biases = new NativeArray<float>(numNeurons, Allocator.Persistent, NativeArrayOptions.ClearMemory);
            Weights = new NativeArray<float>(numNeurons * numInputs, Allocator.Persistent, NativeArrayOptions.ClearMemory);
            Outputs = new NativeArray<float>(numNeurons, Allocator.Persistent, NativeArrayOptions.ClearMemory);
            NumNeurons = numNeurons;
            NumInputs = numInputs;
        }

        public void Dispose() {
            Biases.Dispose();
            Weights.Dispose();
            Outputs.Dispose();
        }
    }

    public class NativeNetwork : System.IDisposable {
        public NativeNetworkLayer[] Layers;

        public NativeNetworkLayer Last {
            get { return Layers[Layers.Length - 1]; }
        }

        public NativeNetworkConfig Config {
            get;
            private set;
        }

        public NativeNetwork(NativeNetworkConfig config) {
            Layers = new NativeNetworkLayer[config.Layers.Count - 1];
            for (int l = 0; l < Layers.Length; l++) {
                Layers[l] = new NativeNetworkLayer(config.Layers[l + 1].Neurons, config.Layers[l].Neurons);
            }
            Config = config;
        }

        public void Dispose() {
            for (int l = 0; l < Layers.Length; l++) {
                Layers[l].Dispose();
            }
        }
    }

    public class NativeGradientsLayer : System.IDisposable {
        public NativeArray<float> DCDZ;
        public NativeArray<float> DCDW;
        public readonly int NumNeurons;
        public readonly int NumInputs;

        public NativeGradientsLayer(int numNeurons, int numInputs) {
            DCDZ = new NativeArray<float>(numNeurons, Allocator.Persistent, NativeArrayOptions.ClearMemory);
            DCDW = new NativeArray<float>(numNeurons * numInputs, Allocator.Persistent, NativeArrayOptions.ClearMemory);
            NumNeurons = numNeurons;
            NumInputs = numInputs;
        }

        public void Dispose() {
            DCDZ.Dispose();
            DCDW.Dispose();
        }
    }

    public class NativeGradients : System.IDisposable {
        public NativeGradientsLayer[] Layers;
        public NativeArray<float> DCDO;
        public NativeNetworkConfig Config;

        public NativeGradientsLayer Last {
            get { return Layers[Layers.Length - 1]; }
        }

        public NativeGradients(NativeNetworkConfig config) {
            Layers = new NativeGradientsLayer[config.Layers.Count - 1];
            for (int l = 0; l < Layers.Length; l++) {
                Layers[l] = new NativeGradientsLayer(config.Layers[l + 1].Neurons, config.Layers[l].Neurons);
            }

            DCDO = new NativeArray<float>(config.Layers[config.Layers.Count - 1].Neurons, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            Config = config;
        }

        public void Dispose() {
            for (int l = 0; l < Layers.Length; l++) {
                Layers[l].Dispose();
            }
            DCDO.Dispose();
        }
    }
}