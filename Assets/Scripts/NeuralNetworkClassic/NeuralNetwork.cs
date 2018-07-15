/*
 * Functional/declarative neural net builder
 * 
 * Todo:
 * - This is an ugly mess now!!!
 * - ILayer abstraction is terrible
 * - Should separate network and optimizer data strctures
 * - Reimplement using Burst/ComputeShaders
 * - Retain support for genetic algorithms, in addition to backprop
 */

using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using Random = System.Random;
using NNClassic;

public enum ActivationType {
    Sigmoid,
    Tanh
}

public enum LayerType {
    Deterministic
}

public struct LayerDefinition {
    public int Size;
    public LayerType LayerType;
    public ActivationType ActivationType;

    public LayerDefinition(int size, LayerType layerType, ActivationType activationType) {
        Size = size;
        LayerType = layerType;
        ActivationType = activationType;
    }

    public override string ToString() {
        return string.Format("Size: {0}, LayerType: {1}, ActivationType: {2}", Size, LayerType, ActivationType);
    }
}

public interface ILayer {
    float[] Forward(float[] input);
    void Backward(ILayer prev, ILayer next);
    void BackwardFinal(ILayer prev, float[] target);

    float[] Outputs { get; }
    float[,] Weights { get; }
    float[,] DCDW { get; }
    float[] Biases { get; }
    float[] DCDZ { get; }
    float[] Z { get; } // Weighted input, used for backpropagation
    int NeuronCount { get; }
    int ParamCount { get; }
    float this[int index] { get; set; }
}

public struct NetDefinition {
    public int InputSize;
    public LayerDefinition[] Layers;

    public NetDefinition(int inputs, params LayerDefinition[] layers) {
        InputSize = inputs;
        Layers = layers;
    }

    public override string ToString() {
        string layers = "";
        for (int i = 0; i < Layers.Length; i++) {
            layers += i + ": " + Layers[i].ToString() + "\n";
        }
        return string.Format("Layers:\n{1}", layers);
    }
}

public class Network {
    public readonly List<ILayer> Layers;

    public float[] Input {
        get { return Layers[0].Outputs; }
    }

    public float[] Output {
        get { return Layers[Layers.Count - 1].Outputs; }
    }

    public Network(List<ILayer> layers) {
        Layers = layers;
    }
}

public static class NetBuilder {
    public static Network Build(NetDefinition definition) {

        List<ILayer> layers = new List<ILayer>();

        layers.Add(new InputLayer(definition.InputSize));

        for (int i = 0; i < definition.Layers.Length; i++) {
            Func<float, float> activation = null;
            Func<float, float> activationD = null;
            switch (definition.Layers[i].ActivationType) {
                case ActivationType.Sigmoid:
                    activation = Utils.Sigmoid;
                    activationD = Utils.SigmoidD;
                    break;
                case ActivationType.Tanh:
                    activation = Utils.Tanh;
                    activationD = null; // Todo:
                    break;
            }

            ILayer l = null;
            switch (definition.Layers[i].LayerType) {
                case LayerType.Deterministic:
                    l = new DeterministicWeightBiasLayer(
                        definition.Layers[i].Size,
                        i == 0 ? definition.InputSize : definition.Layers[i - 1].Size,
                        activation,
                        activationD);
                    break;
            }

            layers.Add(l);
        }

        Network network = new Network(layers);

        return network;
    }
}

public class InputLayer : ILayer {
    private readonly float[] _o;

    public int NeuronCount {
        get { return _o.Length; }
    }

    public int ParamCount {
        get { return 0; }
    }

    public float this[int index] {
        get {
            return 0f;
        }
        set {
            return;
        }
    }

    public float[] Outputs {
        get { return _o; }
    }

    public float[,] Weights {
        get { throw new NotImplementedException(); }
    }

    public float[,] DCDW {
        get { throw new NotImplementedException(); }
    }

    public float[] Biases {
        get { throw new NotImplementedException(); }
    }

    public float[] DCDZ {
        get { throw new NotImplementedException(); }
    }

    public float[] Z {
        get { throw new NotImplementedException(); }
    }

    public InputLayer(int size) {
        _o = new float[size];
    }

    public float[] Forward(float[] input) {
        throw new NotImplementedException();
    }

    public void Backward(ILayer prev, ILayer next) {
        throw new NotImplementedException();
    }
    public void BackwardFinal(ILayer prev, float[] dCdO) {
        throw new NotImplementedException();
    }
}

public class DeterministicWeightBiasLayer : ILayer {
    private readonly float[] _o; // Non-linear activations
    private readonly float[] _z; // Linear activations (kept for backpropagation)
    private readonly Func<float, float> _act; // Activation function
    private readonly Func<float, float> _actD; // 1st derivative of activation function

    // Params
    private readonly float[] _b; // Biases
    private readonly float[] _dCdZ; // Bias gradients
    private readonly float[,] _w; // Weights
    private readonly float[,] _dWdC; // Weight gradients

    /* Abstract parameter access for genetic algorithms
     * Todo: this way of flattening the params is not great, but
     * it has to be done somehow. */

    public int NeuronCount {
        get { return _o.Length; }
    }

    public int ParamCount {
        get { return _b.Length + _w.GetLength(0) * _w.GetLength(1); }
    }

    public float this[int index] {
        get {
            if (index < _b.Length) {
                return _b[index];
            }
            index -= _b.Length;
            return _w[
                index / _w.GetLength(1),
                index % _w.GetLength(1)];
        }
        set {
            if (index < _b.Length) {
                _b[index] = value;
                return;
            }
            index -= _b.Length;
            _w[
                index / _w.GetLength(1),
                index % _w.GetLength(1)] = value;
        }
    }

    public float[] Z {
        get { return _z; }
    }

    public float[] Outputs {
        get { return _o; }
    }

    public float[,] Weights {
        get { return _w; }
    }

    public float[] Biases {
        get { return _b; }
    }

    public float[,] DCDW {
        get { return _dWdC; }
    }

    public float[] DCDZ {
        get { return _dCdZ; }
    }

    public DeterministicWeightBiasLayer(int size, int prevLayerSize, Func<float, float> activation, Func<float, float> activationD) {
        _o = new float[size];
        _z = new float[size];
        _b = new float[size];
        _w = new float[size, prevLayerSize];
        _dCdZ = new float[size];
        _dWdC = new float[size, prevLayerSize];
        _act = activation;
        _actD = activationD;
    }

    public float[] Forward(float[] input) {
        if (_w.GetLength(1) != input.Length) {
            throw new ArgumentException("Number of inputs from last layer does not match number of weights from this layer");
        }

        for (int n = 0; n < _o.Length; n++) {
            // Linear activation
            _z[n] = _b[n];
            for (int w = 0; w < input.Length; w++) {
                _z[n] += input[w] * _w[n, w];
            }
            // Non-linear activation
            _o[n] = _act(_z[n]);
        }

        return _o;
    }

    public void BackwardFinal(ILayer prev, float[] target) {
        float[] dCdO = new float[target.Length];
        NetUtils.Subtract(_o, target, dCdO);

        for (int n = 0; n < _o.Length; n++) {
            float dOdZ = _actD(_o[n]); // Reuses forward pass evaluation of act(z)
            _dCdZ[n] = dCdO[n] * dOdZ;

            for (int w = 0; w < prev.Outputs.Length; w++) {
                _dWdC[n, w] = _dCdZ[n] * prev.Outputs[w];
            }
        }
    }

    public void Backward(ILayer prev, ILayer next) {
        for (int n = 0; n < _o.Length; n++) {
            float dOdZ = _actD(_o[n]);

            _dCdZ[n] = 0f;
            for (int nNext = 0; nNext < next.NeuronCount; nNext++) {
                _dCdZ[n] += next.DCDZ[nNext] * next.Weights[nNext, n];
            }
            _dCdZ[n] *= dOdZ;

            for (int w = 0; w < prev.NeuronCount; w++) {
                _dWdC[n, w] = _dCdZ[n] * prev.Outputs[w];
            }
        }
    }
}

public static class NetUtils {
    public static void LabelToOneHot(int label, float[] vector) {
        for (int i = 0; i < vector.Length; i++) {
            vector[i] = i == label ? 1f : 0f;
        }
    }

    public static void Subtract(float[] a, float[] b, float[] result) {
        if (a.Length != b.Length) {
            throw new System.ArgumentException("Lengths of arrays have to match");
        }

        for (int i = 0; i < a.Length; i++) {
            result[i] = a[i] - b[i];
        }
    }

    public static float Cost(float[] vector) {
        float sum = 0f;
        for (int i = 0; i < vector.Length; i++) {
            sum += vector[i] * vector[i];
        }
        return Mathf.Sqrt(sum);
    }

    public static void Forward(Network net) {
        float[] input = net.Input;
        for (int l = 1; l < net.Layers.Count; l++) {
            input = net.Layers[l].Forward(input);
        }
    }

    public static void Backward(Network net, float[] target) {
        UnityEngine.Profiling.Profiler.BeginSample("Backward");
            
        net.Layers[net.Layers.Count-1].BackwardFinal(net.Layers[net.Layers.Count - 2], target);

        for (int l = net.Layers.Count-2; l > 0; l--) {
            net.Layers[l].Backward(net.Layers[l - 1], net.Layers[l + 1]);
        }
            
        UnityEngine.Profiling.Profiler.EndSample();
    }

    public static void UpdateParameters(Network net, Network gradients, float learningRate) {
        UnityEngine.Profiling.Profiler.BeginSample("UpdateParameters");
            
        for (int l = 1; l < net.Layers.Count; l++) {
            for (int n = 0; n < net.Layers[l].NeuronCount; n++) {
                net.Layers[l].Biases[n] -= gradients.Layers[l].DCDZ[n] * learningRate;

                for (int w = 0; w < net.Layers[l-1].Outputs.Length; w++) {
                    net.Layers[l].Weights[n, w] -= gradients.Layers[l].DCDW[n, w] * learningRate;
                }
            }
        }
            
        UnityEngine.Profiling.Profiler.EndSample();
    }

    public static int GetMaxOutput(Network network) {
        float largestActivation = float.MinValue;
        int idx = 0;
        for (int i = 0; i < network.Output.Length; i++) {
            if (network.Output[i] > largestActivation) {
                largestActivation = network.Output[i];
                idx = i;
            }
        }
        return idx;
    }

    public const float PMax = 1f;

    public static void Zero(Network network) {
        for (int l = 0; l < network.Layers.Count; l++) {
            for (int p = 0; p < network.Layers[l].ParamCount; p++) {
                network.Layers[l][p] = 0f;
            }
        }
    }

    public static void RandomGaussian(Network network, Random r) {
        for (int l = 0; l < network.Layers.Count; l++) {
            for (int p = 0; p < network.Layers[l].ParamCount; p++) {
                network.Layers[l][p] = Utils.Gaussian(r, 0f, 1f);
            }
        }
    }

    public static void Copy(Network fr, Network to) {
        // Todo: If netdefs don't match, error

        for (int l = 0; l < fr.Layers.Count; l++) {
            for (int p = 0; p < fr.Layers[l].ParamCount; p++) {
                to.Layers[l][p] = fr.Layers[l][p];
            }
        }
    }

    public static void Serialize(Network net) {
        // FileStream stream = new FileStream("E:\\code\\unity\\NeuroParty\\Nets\\Test.Net", FileMode.Create);
        // BinaryWriter writer = new BinaryWriter(stream);

        // // Version Info

        // writer.Write(1);

        // // Topology

        // writer.Write(net.Topology.Length);
        // for (int i = 0; i < net.Topology.Length; i++) {
        //     writer.Write(net.Topology[i]);
        // }

        // for (int l = 0; l < net.Topology.Length; l++) {
        //     for (int n = 0; n < net.Topology[l]; n++) {
        //         writer.Write(net.B[l][n]);
        //     }

        //     if (l > 0) {
        //         for (int n = 0; n < net.W[l].Length; n++) {
        //             for (int w = 0; w < net.W[l][n].Length; w++) {
        //                 writer.Write(net.W[l][n][w]);
        //             }
        //         }
        //     }
        // }

        // writer.Close();
    }

    public static Network Deserialize() {
        // FileStream stream = new FileStream("E:\\code\\unity\\NeuroParty\\Nets\\Test.Net", FileMode.Open);
        // BinaryReader reader = new BinaryReader(stream);

        // // Version Info

        // int version = reader.ReadInt32();
        // Debug.Log("Net Version:" + version);

        // // Topology

        // int numLayers = reader.ReadInt32();
        // int[] topology = new int[numLayers];
        // for (int i = 0; i < numLayers; i++) {
        //     topology[i] = reader.ReadInt32();
        // }

        // Network net = new Network(topology);

        // for (int l = 0; l < topology.Length; l++) {
        //     for (int n = 0; n < net.Topology[l]; n++) {
        //         net.B[l][n] = reader.ReadSingle();
        //     }


        //     if (l > 0) {
        //         for (int n = 0; n < net.W[l].Length; n++) {
        //             for (int w = 0; w < net.W[l][n].Length; w++) {
        //                 net.W[l][n][w] = reader.ReadSingle();
        //             }
        //         }
        //     }
        // }

        // reader.Close();

        // return net;
        return null;
    }

    #region Genetic Algorithm

    public static void CrossOver(Network a, Network b, Network c, Random r) {
        for (int l = 0; l < c.Layers.Count; l++) {
            int paramsLeft = c.Layers[l].ParamCount;
            while (paramsLeft > 0) {
                int copySeqLength = Math.Min(r.Next(4, 16), paramsLeft);
                Network parent = r.NextDouble() < 0.5f ? a : b;
                for (int p = 0; p < copySeqLength; p++) {
                    c.Layers[l][p] = parent.Layers[l][p];
                }
                paramsLeft -= copySeqLength;
            }
        }
    }

    public static void Mutate(Network network, float chance, float magnitude, Random r) {
        for (int l = 0; l < network.Layers.Count; l++) {
            for (int p = 0; p < network.Layers[l].ParamCount; p++) {
                if (r.NextDouble() < chance) {
                    network.Layers[l][p] = Mathf.Clamp(network.Layers[l][p] + Utils.Gaussian(r, 0.0, 1.0) * magnitude, -PMax, PMax);
                }
            }
        }
    }

    public static void ScoreBasedWeightedAverage(List<NeuralCreature> nets, Network genotype) {
        float scoreSum = 0f;

        for (int i = 0; i < nets.Count; i++) {
            scoreSum += Mathf.Pow(nets[i].Score, 3f);
        }

        Zero(genotype);

        for (int i = 0; i < nets.Count; i++) {
            float scale = Mathf.Pow(nets[i].Score, 3f) / scoreSum;
            Network candidate = nets[i].Mind;

            for (int l = 0; l < candidate.Layers.Count; l++) {
                for (int p = 0; p < candidate.Layers[l].ParamCount; p++) {
                    genotype.Layers[l][p] += candidate.Layers[l][p] * scale;
                }
            }
        }
    }

    #endregion
}
