using UnityEngine;
using Unity.Burst;
using Unity.Jobs;
using Unity.Collections;
using Unity.Mathematics;
using Rng = Unity.Mathematics.Random;
using Ramjet.Math.FixedPoint;
using System;


/*
    Todo:

    - Information-preserving normalization & scaling strategies to replace Float and Softmax
        - Input set energy analysis

    ===============

    MNIST inputs are 28x28 unsigned bytes, let's attempt to utilize that compactness by
    using neural networks based on integer arithmetic.

    Now let those inputs flow into a layer of neurons.

    ** Weighting **

    qu8_0 * qs0_7 = how many bits of precision? Be careful that we don't lose a tonne
    of information here. For worst case, we need 16 bits of precision in the result
    of each single multiplication. This will then carry over to the neurons, which
    aggregate over these results.

    ** Accumulation **

    Choice time!

    If we make a fully-connected multi-layer perceptron, and we add all weighted
    inputs of 8-bit type, and assuming the worst case of all inputs and weights
    being 1.0, we get a resulting integer value of:

    8 + log2(28*28) = ~17.6147 = 18 bits

    Or with 16 bit weighted inputs:

    16 + log2(28*28) = ~25.6147 = 26 bits

    (This is because each newly tacked-on bit can absorb yet 2^(i-1) extra additions.)

    For now, we could thus have a single neuron sum over the entire image, using a single
    32-bit accumulator. I bet we could use 16-bits or less, because the input images are
    never full white. In fact, we could go over all the input images in the dataset and
    find the worst cose, yielding a solid upper bound on the number of bits needed for
    accumulators.

    Anyway, we sum them, and can arbitrarily reinterpret the scale of the result, if need be.

    === Scaling & Normalization Issue ===

    Floating Point and SoftMax are almost universally applied. Why? Together they form
    a very general solution to normalization issues.

    Input * Network can product wildly varying magnitudes of activations and outputs.
    Softmax  and float are a kind of automatic gain knob on everything that helps to
    preserve important relationships in the values as they travel through the network.

    We'll need something similar to that kind of function, or knowningly choose a
    different approach.

    As mentioned further up, we can analyze the energy values of the input set to find
    much tighter bounds on minimum and maximum expected energy, and design our activation
    functions around this.

    --

    Code generation would be the best approach for creating optimal circuitry here
    Desired functions could be generated on the fly.
 */
public class FixedPointNetwork : MonoBehaviour {
    private NativeArray<byte> _l0_outputs;

    private NativeArray<sbyte> _l1_weights;
    private NativeArray<Int32> _l1_accumulators;
    private NativeArray<byte> _l1_outputs;

    private NativeArray<sbyte> _l2_weights;
    private NativeArray<Int32> _l2_accumulators;
    private NativeArray<byte> _l2_outputs;

    const int _l0_numNeurons = 28 * 28; // img dimensions;
    const int _l1_numNeurons = 30;

    const int _l2_numNeurons = 10; // output dimensions

    private void Awake() {
        // Allocate

        _l0_outputs = new NativeArray<byte>(_l0_numNeurons, Allocator.Persistent);

        _l1_weights = new NativeArray<sbyte>(_l1_numNeurons * _l0_numNeurons, Allocator.Persistent);
        _l1_accumulators = new NativeArray<Int32>(_l1_numNeurons, Allocator.Persistent);
        _l1_outputs = new NativeArray<byte>(_l1_numNeurons, Allocator.Persistent);

        _l2_weights = new NativeArray<sbyte>(_l2_numNeurons * _l1_numNeurons, Allocator.Persistent);
        _l2_accumulators = new NativeArray<Int32>(_l2_numNeurons, Allocator.Persistent);
        _l2_outputs = new NativeArray<byte>(_l2_numNeurons, Allocator.Persistent);

        // Initialize

        var rng = new Rng(1234);
        InitializeWeights(_l1_weights, ref rng);
        InitializeWeights(_l2_weights, ref rng);

        NNBurst.Mnist.DataManager.Load();

        // Input layer

        NNBurst.Mnist.DataManager.CopyBytesToInput(_l0_outputs, NNBurst.Mnist.DataManager.TrainBytes, 0);

        // Hidden layer

        for (int n = 0; n < _l1_accumulators.Length; n++) {
            for (int w = 0; w < _l0_outputs.Length; w++) {
                _l1_accumulators[n] += MulWeight(_l0_outputs[w], _l1_weights[n * _l0_outputs.Length + w]);
            }
        }

        for (int n = 0; n < _l1_accumulators.Length; n++) {
            _l1_outputs[n] = Activate(_l1_accumulators[n]);
        }

        // Output layer

        for (int n = 0; n < _l2_accumulators.Length; n++) {
            for (int w = 0; w < _l1_outputs.Length; w++) {
                _l2_accumulators[n] += MulWeight(_l1_outputs[w], _l2_weights[n * _l1_outputs.Length + w]);
            }
        }

        for (int n = 0; n < _l2_accumulators.Length; n++) {
            _l2_outputs[n] = Activate(_l2_accumulators[n]);
            Debug.Log(n + ": " + _l2_outputs[n]);
        }

        Debug.Log("Predicted label: " + GetHighestOutputIndex(_l2_outputs));
    }

    private void OnDestroy() {
        _l0_outputs.Dispose();

        _l1_accumulators.Dispose();
        _l1_outputs.Dispose();
        _l1_weights.Dispose();

        _l2_accumulators.Dispose();
        _l2_outputs.Dispose();
        _l2_weights.Dispose();
    }

    private static void InitializeWeights(NativeSlice<sbyte> weights, ref Rng rng) {
        // Since we want to assign random bits, the integer type or wordsize doesn't matter
        // So we can reinterpret as generic uints, randomize, and save some compute that way
        var weightInts = weights.SliceConvert<uint>();
        for (int i = 0; i < weightInts.Length; i++) {
            weightInts[i] = rng.NextUInt();
        }
    }

    private static int MulWeight(byte input, sbyte weight) {
        return (input * weight);
    }

    private static byte Activate(Int32 input) {
        /*
        Squash accumulated weighted activations back down to 8 bit value.
        Linear for now, with any negative values being discarded, like a ReLU.

        Assuming that sum of 28x28 8-byte inputs * sbyte weight has at most 26 bits of meaningful precision.

        === Problem ===

        This function is designed according to maximum-possible activation values of the network. However,
        almost all real inputs to the network will produce vastly
        */
        return (byte)(((uint)math.max(0, input)) >> (26-8));
    }

    private static int GetHighestOutputIndex(NativeSlice<byte> output) {
        int highestValue = 0;
        int highestIndex = 0;

        for (int i = 0; i < output.Length; i++) {
            if (output[i] > highestValue) {
                highestValue = output[i];
                highestIndex = i;
            }
        }

        return highestIndex;
    }
}
