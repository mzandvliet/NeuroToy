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

    - Wavelet-like input object representation that models energies in common input image energy bounds
        - Or curve fits, for sure

    - Consider making it logically impossible to pass a zero-energy signal, or any signals over
    max-energy threshold

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


    ---

    Funny... I remember when a few years ago I first studied basic neural network stuff
    I was very bemused with any fussing about input value distributions, preprocessing
    and normalizing it. Looking back, that was so incredibly naive. Yet, crucially,
    I had to discover for myself the information-theoretical reasons for the fuss. And
    now that I have, it looks like we ought to be fussing over it a lot more still.
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
        NNBurst.Mnist.DataManager.LoadByteData();

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

        AnalyzeMnistEnergyBounds();
        ForwardPass();
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

    private static void AnalyzeMnistEnergyBounds() {
        const int ImgPixCount = 28 * 28;

        const int MinEnergyLimit = ImgPixCount * 0;
        const int MaxEnergyLimit = ImgPixCount * 256;

        Debug.Log("Minimum possible energy per image: " + MinEnergyLimit);
        Debug.Log("Maximum possible energy per image: " + MaxEnergyLimit);

        int imgCount = NNBurst.Mnist.DataManager.TrainBytes.NumImgs;
        var imgs = NNBurst.Mnist.DataManager.TrainBytes.Images;

        uint lowestEnergy = uint.MaxValue;
        uint highestEnergy = 0;
        ulong totalEnergy = 0;

        for (int i = 0; i < imgCount; i++) {
            uint energy = 0;
            for (int p = 0; p < ImgPixCount; p++) {
                energy += imgs[i * ImgPixCount + p];
            }

            if (energy > highestEnergy) {
                highestEnergy = energy;
            }
            if (energy < lowestEnergy) {
                lowestEnergy = energy;
            }

            totalEnergy += energy;
        }

        double averageEnergy = totalEnergy / (double)imgCount;

        Debug.LogFormat("Energy bounds: [{0}, {1}]", lowestEnergy, highestEnergy);
        Debug.LogFormat("Average energy: {0}", averageEnergy);

        Debug.LogFormat("As normalized:");
        Debug.LogFormat("Energy bounds: [{0}, {1}]", lowestEnergy / (double)MaxEnergyLimit, highestEnergy / (double)MaxEnergyLimit);
        Debug.LogFormat("Average energy: {0}", averageEnergy / (double)MaxEnergyLimit);

        Debug.LogFormat("As log2:");
        Debug.LogFormat("Energy bounds: [{0}, {1}]", math.log2(lowestEnergy), math.log2(highestEnergy));
        Debug.LogFormat("Average energy: {0}", math.log2(averageEnergy));

        /*
        Outputs:

        Minimum possible energy per image: 0
        Maximum possible energy per image: 200704

        Energy bounds: [5086, 79483]
        Average energy: 26121,6424166667

        As normalized:
        Energy bounds: [0,0253408003826531, 0,396021006058673]
        Average energy: 0,130150083788398

        As log2:
        Energy bounds: [12,31232, 16,27836]
        Average energy: 14,6729579897526

        ----

        We can see that while any pixel can peak to 255, expected
        total energies for Mnist inputs tends to be very low. 

        Indeed, most pixels will remain fully black for any image.

        Our computational process should be able to reflect this
        sparseness.

        This also helps us design our activation and scaling functions.
        */
    }

    private void ForwardPass() {
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
