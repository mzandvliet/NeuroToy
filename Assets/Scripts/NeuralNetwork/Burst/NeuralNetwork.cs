

using System.Collections.Generic;
using Unity.Collections;

namespace NNBurst {
    public static class NeuralUtils {
        public static void Initialize(FCNetwork net, System.Random random) {
            // Todo: init as jobs too. Needs Burst-compatible RNG.

            for (int l = 0; l < net.Layers.Length; l++) {
                NeuralMath.RandomGaussian(random, net.Layers[l].Weights, 0f, 1f);
                NeuralMath.RandomGaussian(random, net.Layers[l].Biases, 0f, 1f);
            }
        }
    }

    #region MLP

    public interface ILayer {

    }

    public struct FCLayerConfig {
        public int Neurons;
    }

    public class FCNetworkConfig {
        public List<FCLayerConfig> Layers;

        public FCNetworkConfig() {
            Layers = new List<FCLayerConfig>();
        }
    }

    // Todo: could test layers just existing of slices of single giant arrays
    public class FCLayer : System.IDisposable {
        public NativeArray<float> Biases;
        public NativeArray<float> Weights;
        public NativeArray<float> Outputs;
        public readonly int NumNeurons;
        public readonly int NumInputs;

        public FCLayer(int numNeurons, int numInputs) {
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

    public class FCGradientsLayer : System.IDisposable {
        public NativeArray<float> DCDZ;
        public NativeArray<float> DCDW;
        public readonly int NumNeurons;
        public readonly int NumInputs;

        public FCGradientsLayer(int numNeurons, int numInputs) {
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

    public class FCNetwork : System.IDisposable {
        public FCLayer[] Layers;

        public FCLayer Last {
            get { return Layers[Layers.Length - 1]; }
        }

        public FCNetworkConfig Config {
            get;
            private set;
        }

        public FCNetwork(FCNetworkConfig config) {
            Layers = new FCLayer[config.Layers.Count - 1];
            for (int l = 0; l < Layers.Length; l++) {
                Layers[l] = new FCLayer(config.Layers[l + 1].Neurons, config.Layers[l].Neurons);
            }
            Config = config;
        }

        public void Dispose() {
            for (int l = 0; l < Layers.Length; l++) {
                Layers[l].Dispose();
            }
        }
    }

    public class FCGradients : System.IDisposable {
        public FCGradientsLayer[] Layers;
        public NativeArray<float> DCDO;
        public FCNetworkConfig Config;

        public FCGradientsLayer Last {
            get { return Layers[Layers.Length - 1]; }
        }

        public FCGradients(FCNetworkConfig config) {
            Layers = new FCGradientsLayer[config.Layers.Count - 1];
            for (int l = 0; l < Layers.Length; l++) {
                Layers[l] = new FCGradientsLayer(config.Layers[l + 1].Neurons, config.Layers[l].Neurons);
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

    #endregion
}