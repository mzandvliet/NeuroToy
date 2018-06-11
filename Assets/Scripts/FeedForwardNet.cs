using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;
using Random = System.Random;

/* 
 * Simple implementation of a classic feed forward network.
 */

public class FeedForwardNetwork {
    private readonly float[][] _a; // Activations [layer, neuron]
    private readonly float[][] _b; // Biases [layer, neuron]
    private readonly float[][][] _w; // Weights [layer, neuron, weight]
    public readonly int[] Topology; // Number of neurons per layer

    // Easy access to input layer for geting sensor data into the net
    public float[] Input {
        get {
            return _a[0];
        }
    }

    // Easy access to output
    public float[] Output {
        get { return _a[_a.Length - 1]; }
    }

    public float[][] A {
        get { return _a; }
    }

    public float[][] B {
        get { return _b; }
    }

    public float[][][] W {
        get { return _w; }
    }

    public FeedForwardNetwork(int[] topology) {
        if (topology == null || topology.Length == 0) {
            throw new ArgumentException("Topology is invalid");
        }

        Topology = topology;

        _a = new float[topology.Length][];
        _b = new float[topology.Length][];
        _w = new float[topology.Length][][];

        for (int l = 0; l < topology.Length; l++) {
            _a[l] = new float[topology[l]];
            _b[l] = new float[topology[l]];

            if (l > 0) {
                _w[l] = new float[topology[l]][];
                for (int n = 0; n < topology[l]; n++) {
                    _w[l][n] = new float[topology[l - 1]];
                }
            }

            for (int n = 0; n < Topology[l]; n++) {
                _a[l][n] = -1234f;
                _b[l][n] = -12345f;
            }

            if (l > 0) {
                for (int n = 0; n < _w[l].Length; n++) {
                    for (int w = 0; w < _w[l][n].Length; w++) {
                        _w[l][n][w] = -123f;
                    }
                }
            }
        }
    }
}

public static class NNUtil {
    public const float WMax = 1f;
    public const float BMax = 1f;

    public static void Copy(FeedForwardNetwork fr, FeedForwardNetwork to) {
        for (int l = 0; l < to.Topology.Length; l++) {
            for (int n = 0; n < to.Topology[l]; n++) {
                to.B[l][n] = fr.B[l][n];
            }
            

            if (l > 0) {
                for (int n = 0; n < to.W[l].Length; n++) {
                    for (int w = 0; w < to.W[l][n].Length; w++) {
                        to.W[l][n][w] = fr.W[l][n][w];
                    }
                }
            }
        }
    }

    public static void CrossOver(FeedForwardNetwork a, FeedForwardNetwork b, FeedForwardNetwork c, Random r) {
        for (int l = 0; l < c.Topology.Length; l++) {
            int neuronsLeft = c.Topology[l];
            while (neuronsLeft > 0) {
                int copySeqLength = Math.Min(r.Next(4, 16), neuronsLeft);
                FeedForwardNetwork parent = r.NextDouble() < 0.5f ? a : b;
                for (int n = 0; n < copySeqLength; n++) {
                    c.B[l][n] = parent.B[l][n];
                }
                neuronsLeft -= copySeqLength;
            }
            
            if (l > 0) {
                for (int n = 0; n < c.W[l].Length; n++) {
                    int weightsLeft = c.W[l][n].Length;
                    while (weightsLeft > 0) {
                        int copySeqLength = Math.Min(r.Next(4, 16), weightsLeft);
                        FeedForwardNetwork parent = r.NextDouble() < 0.5f ? a : b;
                        for (int w = 0; w < copySeqLength; w++) {
                            c.W[l][n][w] = parent.W[l][n][w];
                        }
                        weightsLeft -= copySeqLength;
                    }
                }
            }
        }
    }

    public static void Mutate(FeedForwardNetwork net, float chance, float magnitude, Random r) {
        for (int l = 0; l < net.Topology.Length; l++) {
            for (int n = 0; n < net.Topology[l]; n++) {
                if (r.NextDouble() < chance) {
                    net.B[l][n] = Mathf.Clamp(net.B[l][n] + Utils.Gaussian(r) * magnitude, -BMax, BMax);
                }
            }

            if (l > 0) {
                for (int n = 0; n < net.W[l].Length; n++) {
                    for (int w = 0; w < net.W[l][n].Length; w++) {
                        if (r.NextDouble() < chance) {
                            net.W[l][n][w] = Mathf.Clamp(net.W[l][n][w] + Utils.Gaussian(r) * magnitude, -WMax, WMax);
                        }
                    }
                }
            }
        }
    }
    
//    public static void ScoreBasedWeightedAverage(List<NeuralCreature> nets, NeuralNetwork genotype) {
//        float scoreSum = 0f;
//
//        for (int i = 0; i < nets.Count; i++) {
//            scoreSum += Mathf.Pow(nets[i].Score, 2f);
//        }
//
//        Zero(genotype);
//
//        for (int i = 0; i < nets.Count; i++) {
//            float scale = Mathf.Pow(nets[i].Score, 2f) / scoreSum;
//            NeuralNetwork candidate = nets[i].Mind;
//            for (int l = 0; l < genotype.Topology.Length; l++) {
//                for (int n = 0; n < genotype.Topology[l]; n++) {
//                    genotype.B[l][n] += candidate.B[l][n] * scale;
//                }
//
//                if (l > 0) {
//                    for (int n = 0; n < genotype.W[l].Length; n++) {
//                        for (int w = 0; w < genotype.W[l][n].Length; w++) {
//                            genotype.W[l][n][w] += candidate.W[l][n][w] * scale;
//                        }
//                    }
//                }
//            }
//        }
//    }

    public static void Randomize(FeedForwardNetwork net, Random r) {
        for (int l = 0; l < net.Topology.Length; l++) {
            for (int n = 0; n < net.Topology[l]; n++) {
                net.B[l][n] = Utils.Gaussian(r) * BMax;
                net.A[l][n] = 0.123456f;
            }

            if (l > 0) {
                for (int n = 0; n < net.W[l].Length; n++) {
                    for (int w = 0; w < net.W[l][n].Length; w++) {
                        net.W[l][n][w] = Utils.Gaussian(r) * WMax;
                    }
                }
            }
        }
    }

    public static void Zero(FeedForwardNetwork net) {
        for (int l = 0; l < net.Topology.Length; l++) {
            for (int n = 0; n < net.Topology[l]; n++) {
                net.B[l][n] = 0f;
            }

            if (l > 0) {
                for (int n = 0; n < net.W[l].Length; n++) {
                    for (int w = 0; w < net.W[l][n].Length; w++) {
                        net.W[l][n][w] = 0f;
                    }
                }
            }
        }
    }

    public static void PropagateForward(FeedForwardNetwork net) {
        for (int l = 1; l < net.Topology.Length; l++) {
            for (int n = 0; n < net.Topology[l]; n++) {
                net.A[l][n] = 0f;
                for (int w = 0; w < net.Topology[l-1]; w++) {
                    float synAct = net.A[l - 1][w] * net.W[l][n][w];
                    net.A[l][n] += synAct;
                }

                net.A[l][n] = Utils.Tanh(net.A[l][n] + net.B[l][n]);
            }
        }
    }

    public static void Serialize(FeedForwardNetwork net) {
        FileStream stream = new FileStream("E:\\code\\unity\\NeuroParty\\Nets\\Test.Net", FileMode.Create);
        BinaryWriter writer = new BinaryWriter(stream);

        // Version Info

        writer.Write(1);

        // Topology

        writer.Write(net.Topology.Length);
        for (int i = 0; i < net.Topology.Length; i++) {
            writer.Write(net.Topology[i]);
        }

        for (int l = 0; l < net.Topology.Length; l++) {
            for (int n = 0; n < net.Topology[l]; n++) {
                writer.Write(net.B[l][n]);
            }
            
            if (l > 0) {
                for (int n = 0; n < net.W[l].Length; n++) {
                    for (int w = 0; w < net.W[l][n].Length; w++) {
                        writer.Write(net.W[l][n][w]);
                    }
                }
            }
        }

        writer.Close();
    }

    public static FeedForwardNetwork Deserialize() {
        FileStream stream = new FileStream("E:\\code\\unity\\NeuroParty\\Nets\\Test.Net", FileMode.Open);
        BinaryReader reader = new BinaryReader(stream);

        // Version Info

        int version = reader.ReadInt32();
        Debug.Log("Net Version:" + version);

        // Topology

        int numLayers = reader.ReadInt32();
        int[] topology = new int[numLayers];
        for (int i = 0; i < numLayers; i++) {
            topology[i] = reader.ReadInt32();
        }

        FeedForwardNetwork net = new FeedForwardNetwork(topology);

        for (int l = 0; l < topology.Length; l++) {
            for (int n = 0; n < net.Topology[l]; n++) {
                net.B[l][n] = reader.ReadSingle();
            }
            

            if (l > 0) {
                for (int n = 0; n < net.W[l].Length; n++) {
                    for (int w = 0; w < net.W[l][n].Length; w++) {
                        net.W[l][n][w] = reader.ReadSingle();
                    }
                }
            }
        }

        reader.Close();

        return net;
    }
}