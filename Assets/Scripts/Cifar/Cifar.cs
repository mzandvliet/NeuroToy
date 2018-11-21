using System.IO;
using UnityEngine;
using Unity.Collections;
using System.Collections.Generic;
using Unity.Jobs;
using Rng = Unity.Mathematics.Random;

namespace NNBurst.Cifar {
    public struct Dataset : System.IDisposable {
        public NativeArray<Label> Labels;
        public NativeArray<float> Images;
        public List<int> Indices;

        public int NumImgs {
            get { return Labels.Length; }
        }

        public Dataset(int count) {
            Labels = new NativeArray<Label>(count, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            Images = new NativeArray<float>(count * DataManager.Height * DataManager.Width * DataManager.Channels, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            Indices = new List<int>(count);
        }

        public void Dispose() {
            Labels.Dispose();
            Images.Dispose();
        }
    }

    public enum Label : byte {
        Airplane,
        Automobile,
        Bird,
        Cat,
        Deer,
        Dog,
        Frog,
        Horse,
        Ship,
        Truck
    }

    public static class DataManager {
        private const string Folder = "./Datasets/Cifar/cifar-10-batches-bin";
        private const string TrainImagePathPrefix = "data_batch_"; // 1 2,3,4,5 .bin
        private const string TrainImagePathPostfix = ".bin";
        private const string TestImagePath = "test_batch.bin";

        const int NumImgsPerFile = 10000;
        public const int Height = 32;
        public const int Width = 32;
        public const int Channels = 3;
        public const int ImgDims = Height * Width;

        public static Dataset Train;
        public static Dataset Test;
        

        public static void Load() {
            Train = new Dataset(NumImgsPerFile * 5);
            for (int i = 0; i < 5; i++) {
                string path = TrainImagePathPrefix + (i+1) + TrainImagePathPostfix;
                Load(path, Train, i * NumImgsPerFile);
            }

            Test = new Dataset(NumImgsPerFile);
            Load(TestImagePath, Test, 0);
        }

        public static void Unload() {
            Debug.Log("CIFAR: Unloading datasets...");
            Train.Dispose();
            Test.Dispose();
        }

        private static void Load(string imgPath, Dataset set, int imgOffset) {
            var imgReader = new BinaryReader(new FileStream(Path.Combine(Folder, imgPath), FileMode.Open));

            Debug.Log("CIFAR-10: Loading " + imgPath);

            // Todo: storing colors in interleaved way probably makes our lives easier

            for (int i = 0; i < NumImgsPerFile; i++) {
                byte lbl = imgReader.ReadByte();
                set.Labels[imgOffset + i] = (Label)lbl;

                int imgStart = ((imgOffset + i) * Channels * ImgDims);
                for (int c = 0; c < Channels; c++) {
                    int colStart = c * ImgDims;
                    for (int p = 0; p < ImgDims; p++) {
                        byte val = imgReader.ReadByte();
                        set.Images[imgStart + colStart + p] = val / 256f;
                    }
                }
            }
        }

        public static void GetBatch(NativeArray<int> batch, Dataset set, ref Rng rng) {
            // Todo: can transform dataset to create additional variation

            UnityEngine.Profiling.Profiler.BeginSample("GetBatch");

            if (set.Indices.Count < batch.Length) {
                set.Indices.Clear();
                for (int i = 0; i < set.NumImgs; i++) {
                    set.Indices.Add(i);
                }
                Ramjet.Utils.Shuffle(set.Indices, ref rng);
            }

            for (int i = 0; i < batch.Length; i++) {
                batch[i] = set.Indices[set.Indices.Count - 1];
                set.Indices.RemoveAt(set.Indices.Count - 1);
            }

            UnityEngine.Profiling.Profiler.EndSample();
        }

        public static JobHandle CopyInput(NativeArray<float> inputs, NNBurst.Cifar.Dataset set, int imgIdx, JobHandle handle = new JobHandle()) {
            var copyInputJob = new CopySubsetJob();
            copyInputJob.From = set.Images;
            copyInputJob.To = inputs;
            copyInputJob.Length = ImgDims * Channels;
            copyInputJob.FromStart = imgIdx * ImgDims * Channels;
            copyInputJob.ToStart = 0;
            return copyInputJob.Schedule(handle);
        }

        public static void ToTexture(Dataset set, int imgIndex, Texture2D tex) {
            var colors = new Color[ImgDims];

            int imgStart = imgIndex * ImgDims * 3;

            for (int y = 0; y < Width; y++) {
                for (int x = 0; x < Width; x++) {
                    int p = y * Height + x;
                    
                    float r = set.Images[imgStart + 1024 * 0 + p];
                    float g = set.Images[imgStart + 1024 * 1 + p];
                    float b = set.Images[imgStart + 1024 * 2 + p];

                    // Invert y
                    colors[(Height - 1 - y) * Height + x] = new Color(r, g, b, 1f);
                }
            }

            tex.SetPixels(0, 0, Width, DataManager.Height, colors);
            tex.Apply(false);
        }
    }
}