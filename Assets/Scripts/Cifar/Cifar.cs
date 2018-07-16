using System.IO;
using UnityEngine;
using Unity.Collections;
using System.Collections.Generic;

namespace NNBurst.Cifar {
    public struct Dataset : System.IDisposable {
        public NativeArray<Label> Labels;
        public NativeArray<float> Images;
        public List<int> Indices;

        public int NumImgs {
            get { return Labels.Length; }
        }

        public int Rows {
            get;
            private set;
        }
        public int Cols {
            get;
            private set;
        }
        public int ImgDims {
            get { return Rows * Cols; }
        }

        public Dataset(int count, int rows, int cols) {
            Labels = new NativeArray<Label>(count, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            Images = new NativeArray<float>(count * rows * cols * 3, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            Indices = new List<int>(count);
            Rows = rows;
            Cols = cols;
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
        const int Rows = 32;
        const int Cols = 32;
        const int Channnels = 3;
        const int ImgDims = Rows * Cols;

        public static Dataset Train;
        public static Dataset Test;
        

        public static void Load() {
            Train = new Dataset(NumImgsPerFile * 5, Rows, Cols);
            for (int i = 0; i < 5; i++) {
                string path = TrainImagePathPrefix + (i+1) + TrainImagePathPostfix;
                Load(path, Train, i * NumImgsPerFile);
            }

            Test = new Dataset(NumImgsPerFile, Rows, Cols);
            Load(TestImagePath, Test, 0);
        }

        public static void Unload() {
            Debug.Log("MNIST: Unloading datasets...");
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

                int imgStart = ((imgOffset + i) * Channnels * ImgDims);
                for (int c = 0; c < Channnels; c++) {
                    int colStart = c * ImgDims;
                    for (int p = 0; p < ImgDims; p++) {
                        byte val = imgReader.ReadByte();
                        set.Images[imgStart + colStart + p] = val / 256f;
                    }
                }
            }
        }

        public static void GetBatch(NativeArray<int> batch, Dataset set, System.Random r) {
            // Todo: can transform dataset to create additional variation

            UnityEngine.Profiling.Profiler.BeginSample("GetBatch");

            if (set.Indices.Count < batch.Length) {
                set.Indices.Clear();
                for (int i = 0; i < set.NumImgs; i++) {
                    set.Indices.Add(i);
                }
                Shuffle(set.Indices, r);
            }

            for (int i = 0; i < batch.Length; i++) {
                batch[i] = set.Indices[set.Indices.Count - 1];
                set.Indices.RemoveAt(set.Indices.Count - 1);
            }

            UnityEngine.Profiling.Profiler.EndSample();
        }

        public static void Shuffle<T>(IList<T> list, System.Random r) {
            int n = list.Count;
            while (n > 1) {
                n--;
                int k = r.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }

        public static void ToTexture(Dataset set, int imgIndex, Texture2D tex) {
            var colors = new Color[set.ImgDims];

            int imgStart = imgIndex * set.ImgDims * 3;

            for (int y = 0; y < set.Cols; y++) {
                for (int x = 0; x < set.Rows; x++) {
                    int p = y * set.Cols + x;
                    
                    float r = set.Images[imgStart + 1024 * 0 + p];
                    float g = set.Images[imgStart + 1024 * 1 + p];
                    float b = set.Images[imgStart + 1024 * 2 + p];

                    // Invert y
                    colors[(set.Cols - 1 - y) * set.Cols + x] = new Color(r, g, b, 1f);
                }
            }

            tex.SetPixels(0, 0, set.Rows, set.Cols, colors);
            tex.Apply(false);
        }
    }
}