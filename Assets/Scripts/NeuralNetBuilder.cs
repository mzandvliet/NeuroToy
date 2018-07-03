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
    void BackwardFinal(ILayer prev, float[] dCdO);

    float[] Outputs { get; }
    float[,] Weights { get; }
    float[,] DCDW { get; }
    float[] Biases { get; }
    float[] DCDZ { get; }
    float[] Z { get; } // Weighted input, used for backpropagation
    int Count { get; }
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
    private readonly float[] _a;

    public int Count {
        get { return _a.Length; }
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
        get { return _a; }
    }

    public float[,] Weights {
        get { return null; }
    }

    public float[,] DCDW {
        get { return null; }
    }

    public float[] Biases {
        get { return null; }
    }

    public float[] DCDZ {
        get { return null; }
    }

    public float[] Z {
        get { return null; }
    }

    public InputLayer(int size) {
        _a = new float[size];
    }

    public float[] Forward(float[] input) {
        for (int i = 0; i < input.Length; i++) {
            _a[i] = input[i];
        }

        return _a;
    }

    public void Backward(ILayer prev, ILayer next) {
    }
    public void BackwardFinal(ILayer prev, float[] dCdO) {
    }
}

public class DeterministicWeightBiasLayer : ILayer {
    private readonly float[] _o; // Non-linear activations
    private readonly float[] _z; // Linear activations (kept for backpropagation)
    private readonly Func<float, float> _act;
    private readonly Func<float, float> _actD;

    // Params
    private readonly float[] _b; // Biases
    private readonly float[] _dCdZ; // Biases gradients
    private readonly float[,] _w; // Weights
    private readonly float[,] _dWdC; // Weight gradients

    /* Abstract parameter access for genetic algorithms
     * Todo: this way of flattening the params is not great, but
     * it has to be done somehow. */

    public int Count {
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
            float a = 0f;
            // Dot inputs with weights
            for (int w = 0; w < input.Length; w++) {
                a += input[w] * _w[n, w];
            }
            // Linear activation is the above plus bias
            _z[n] = a + _b[n];
            // Non-linear activation
            _o[n] = _act(_z[n]);
        }

        return _o;
    }

    public void BackwardFinal(ILayer prev, float[] dCdO) {
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
            for (int n2 = 0; n2 < next.Outputs.Length; n2++) {
                _dCdZ[n] = next.DCDW[n2, n] * dOdZ * dOdZ;
            }

            for (int w = 0; w < prev.Outputs.Length; w++) {
                _dWdC[n, w] = _dCdZ[n] * prev.Outputs[w];
            }
        }
    }
}

public static class NetUtils {
    public static void Forward(Network net) {
        float[] input = net.Input;
        for (int l = 1; l < net.Layers.Count; l++) {
            input = net.Layers[l].Forward(input);
        }
    }

    public static void Backward(Network net, float[] target) {
        float[] dCdO = new float[target.Length];
        Mnist.Subtract(net.Output, target, dCdO);

        net.Layers[net.Layers.Count-1].BackwardFinal(net.Layers[net.Layers.Count - 2], dCdO);

        for (int l = net.Layers.Count-2; l > 0; l--) {
            net.Layers[l].Backward(net.Layers[l - 1], net.Layers[l + 1]);
        }
    }

    public static void UpdateParameters(Network net, Network gradients, float learningRate) {
        for (int l = 1; l < net.Layers.Count; l++) {
            for (int n = 0; n < net.Layers[l].Count; n++) {
                net.Layers[l].Biases[n] -= gradients.Layers[l].DCDZ[n] * learningRate;

                for (int w = 0; w < net.Layers[l-1].Outputs.Length; w++) {
                    net.Layers[l].Weights[n, w] -= gradients.Layers[l].DCDW[n, w] * learningRate;
                }
            }
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
