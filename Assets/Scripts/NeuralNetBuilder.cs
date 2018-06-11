/*
 * Functional/declarative neural net builder
 * 
 * A net = an ordered set of layers
 * 
 * A layer = 
 *   - neurons, with sets of params (type can vary, but all are a bag-of-floats)
 *   - activation type (the function that aggregates connections into final value)
 *   
 * This setup will allow us to mix and match layer types, such as:
 * - Using both deterministic and stochastic layers
 * - Using different activation functions
 */

/* Todo:
 * - Reimplement using Burst/ComputeShaders
 * - Offer abstract bag-of-floats interface to optimizer
 * - Offer abstract input output mapping interface to evaluator
 * - Leave the rest of the interface black box
 * 
 * This will allow us to write optimizers that work on any neural net or policy
 * that is parameterized by floats
 */

using System;
using System.Collections.Generic;
using UnityEngine;
using Random = System.Random;

public enum ActivationType {
    Sigmoid,
    Tanh
}

public enum LayerType {
    Deterministic,
    Stochastic
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
    float[] Activations { get; }
    float[] Z { get; } // Weighted input, used for backpropagation
    int ParamCount { get; }
    float this[int index] { get; set; }
}

public struct NetDefinition {
    public int Inputs;
    public LayerDefinition[] Layers;

    public NetDefinition(int inputs, params LayerDefinition[] layers) {
        Inputs = inputs;
        Layers = layers;
    }

    public override string ToString() {
        string layers = "";
        for (int i = 0; i < Layers.Length; i++) {
            layers += i + ": " + Layers[i].ToString() + "\n";
        }
        return string.Format("Inputs: {0}, Layers:\n{1}", Inputs, layers);
    }
}

public class Network {
    private readonly float[] _input;
    public readonly List<ILayer> Layers;

    public float[] Input {
        get { return _input; }
    }

    public float[] Output {
        get { return Layers[Layers.Count - 1].Activations; }
    }

    public Network(int inputs, List<ILayer> layers) {
        _input = new float[inputs];
        Layers = layers;
    }
}

public static class NetBuilder {
    public static Network Build(NetDefinition definition) {

        List<ILayer> layers = new List<ILayer>();

        for (int i = 0; i < definition.Layers.Length; i++) {
            Func<float, float> activation = null;
            switch (definition.Layers[i].ActivationType) {
                case ActivationType.Sigmoid:
                    activation = Utils.Sigmoid;
                    break;
                case ActivationType.Tanh:
                    activation = Utils.Tanh;
                    break;
            }

            ILayer l = null;
            switch (definition.Layers[i].LayerType) {
                case LayerType.Deterministic:
                    l = new DeterministicWeightBiasLayer(
                        definition.Layers[i].Size,
                        i == 0 ? definition.Inputs : definition.Layers[i-1].Size,
                        activation);
                    break;
                case LayerType.Stochastic:
                    l = new StochasticWeightBiasLayer(
                        definition.Layers[i].Size,
                        i == 0 ? definition.Inputs : definition.Layers[i - 1].Size,
                        activation);
                    break;
            }

            layers.Add(l);
        }

        Network network = new Network(definition.Inputs, layers);

        return network;
    }
}

public class DeterministicWeightBiasLayer : ILayer {
    private readonly float[] _a;
    private readonly float[] _z;
    private readonly Func<float, float> _act;

    // Params
    private readonly float[] _b;
    private readonly float[,] _w;

    /* Abstract parameter access
     * Todo: this way of flattening the params is not great, but
     * it has to be done somehow. */

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

    public float[] Activations {
        get { return _a; }
    }

    public float[] Z {
        get { return _z; }
    }

    public DeterministicWeightBiasLayer(int size, int inputs, Func<float, float> activation) {
        _a = new float[size];
        _z = new float[size];
        _b = new float[size];
        _w = new float[size, inputs];
        _act = activation;
    }

    public float[] Forward(float[] input) {
        if (_w.GetLength(1) != input.Length) {
            throw new ArgumentException("Number of inputs from last layer does not match number of weights from this layer");
        }

        for (int n = 0; n < _a.Length; n++) {
            float a = 0f;
            for (int w = 0; w < input.Length; w++) {
                a += input[w] * _w[n, w];
            }
            _z[n] = a + _b[n];
            _a[n] = _act(_z[n]);
        }

        return _a;
    }
}

public class StochasticWeightBiasLayer : ILayer {
    private readonly float[] _a;
    private readonly float[] _z;
    private readonly Func<float, float> _act;

    // Params
    private readonly float[,] _b;
    private readonly float[,] _w;

    /* Abstract parameter access
     * Todo: this way of flattening the params is not great, but
     * it works for the genetic algorithm mutations. */

    public int ParamCount {
        get { return _b.Length + _w.GetLength(0) * _w.GetLength(1); }
    }

    public float this[int index] {
        get {
            if (index < _b.Length) {
                return _b[
                    index / _b.GetLength(1),
                    index % _b.GetLength(1)];
            }

            index -= _b.Length;
            return _w[
                index / _w.GetLength(1),
                index % _w.GetLength(1)];
        }
        set {
            if (index < _b.Length) {
                _b[
                    index / _b.GetLength(1),
                    index % _b.GetLength(1)] = value;
                return;
            }

            index -= _b.Length;
            _w[
                index / _w.GetLength(1),
                index % _w.GetLength(1)] = value;
        }
    }

    public float[] Activations {
        get { return _a; }
    }

    public float[] Z {
        get { return _z; }
    }

    public StochasticWeightBiasLayer(int size, int inputs, Func<float, float> activation) {
        _a = new float[size];
        _z = new float[size];
        _b = new float[size, 2];
        _w = new float[size, inputs];
        _act = activation;
    }

    public float[] Forward(float[] input) {
        if (_w.GetLength(1) != input.Length) {
            throw new ArgumentException("Number of inputs from last layer does not match number of weights from this layer");
        }

        for (int n = 0; n < _a.Length; n++) {
            float a = 0f;
            for (int w = 0; w < input.Length; w++) {
                //float wval = Utils.Gaussian(_w[n, w], 0.1f); // Todo: parameterize
                a += input[w] * _w[n, w];
            }
            _z[n] = a + Utils.Gaussian(_b[n, 0], _b[n, 1]);
            _a[n] = _act(_z[n]);
        }

        return _a;
    }
}

public static class NetUtils {
    public static void Forward(Network network) {
        float[] input = network.Input;
        for (int i = 0; i < network.Layers.Count; i++) {
            input = network.Layers[i].Forward(input);
        }
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

    public static void Backward(Network network) {
        // Todo: implement
    }

    public const float WMax = 1f;
    public const float PMax = 1f;

    #region Genetic Algorithm

    public static void Copy(Network fr, Network to) {
        // Todo: If netdefs don't match, error

        for (int l = 0; l < fr.Layers.Count; l++) {
            for (int p = 0; p < fr.Layers[l].ParamCount; p++) {
                to.Layers[l][p] = fr.Layers[l][p];
            }
        }
    }

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
                    network.Layers[l][p] = Mathf.Clamp(network.Layers[l][p] + Utils.Gaussian(r) * magnitude, -PMax, PMax);
                }
            }
        }
    }

    public static void Zero(Network network) {
        for (int l = 0; l < network.Layers.Count; l++) {
            for (int p = 0; p < network.Layers[l].ParamCount; p++) {
                network.Layers[l][p] = 0f;
            }
        }
    }

    public static void Randomize(Network network, Random r) {
        for (int l = 0; l < network.Layers.Count; l++) {
            for (int p = 0; p < network.Layers[l].ParamCount; p++) {
                network.Layers[l][p] = Utils.Gaussian(r);
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
