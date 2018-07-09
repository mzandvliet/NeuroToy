using System.IO;
using UnityEngine;
using Unity.Collections;
using System.Collections.Generic;

namespace New {
    public struct Dataset : System.IDisposable {
        public NativeArray<int> Labels;
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
            Labels = new NativeArray<int>(count, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            Images = new NativeArray<float>(count * rows * cols, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            Indices = new List<int>(count);
            Rows = rows;
            Cols = cols;
        }

        public void Dispose() {
            Labels.Dispose();
            Images.Dispose();
        }
    }

    public static class Mnist {
        private const string Folder = "Datasets\\Mnist";
        private const string TrainImagePath = "train-images.idx3-ubyte";
        private const string TrainLabelPath = "train-labels.idx1-ubyte";
        private const string TestImagePath = "t10k-images.idx3-ubyte";
        private const string TestLabelPath = "t10k-labels.idx1-ubyte";

        public static Dataset Train;
        public static Dataset Test;

        public static void Load() {
            Train = Load(TrainImagePath, TrainLabelPath);
            Test = Load(TestImagePath, TestLabelPath);
        }

        public static void Unload() {
            Debug.Log("MNIST: Unloading datasets...");
            Train.Dispose();
            Test.Dispose();
        }

        private static Dataset Load(string imgPath, string lblPath) {
            var lblReader = new BigEndianBinaryReader(new FileStream(Path.Combine(Folder, lblPath), FileMode.Open));
            var imgReader = new BigEndianBinaryReader(new FileStream(Path.Combine(Folder, imgPath), FileMode.Open));

            lblReader.ReadInt32();
            lblReader.ReadInt32();

            int magicNum = imgReader.ReadInt32();
            int NumImgs = imgReader.ReadInt32();
            int Rows = imgReader.ReadInt32();
            int Cols = imgReader.ReadInt32();
            int ImgDims = Rows * Cols;

            Debug.Log("MNIST: Loading " + imgPath + ", Imgs: " + NumImgs + " Rows: " + Rows + " Cols: " + Cols);

            var set = new Dataset(NumImgs, Rows, Cols);

            for (int i = 0; i < NumImgs; i++) {
                byte lbl = lblReader.ReadByte();
                set.Labels[i] = (int)lbl;
            }

            for (int i = 0; i < NumImgs; i++) {
                for (int j = 0; j < ImgDims; j++) {
                    byte pix = imgReader.ReadByte();
                    set.Images[i * ImgDims + j] = pix / 256f;
                }
            }

            return set;
        }

        public static Batch GetBatch(int size, Dataset set, System.Random r) {
            // Todo: can transform dataset to create additional variation

            UnityEngine.Profiling.Profiler.BeginSample("GetBatch");

            if (set.Indices.Count < size) {
                set.Indices.Clear();
                for (int i = 0; i < set.NumImgs; i++) {
                    set.Indices.Add(i);
                }
                Shuffle(set.Indices, r);
            }

            Batch b = new Batch(size);
            for (int i = 0; i < size; i++) {
                b.Indices[i] = set.Indices[set.Indices.Count - 1];
                set.Indices.RemoveAt(set.Indices.Count - 1);
            }

            UnityEngine.Profiling.Profiler.EndSample();

            return b;
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

            for (int y = 0; y < set.Cols; y++) {
                for (int x = 0; x < set.Rows; x++) {
                    float pix = set.Images[imgIndex * set.ImgDims + y * set.Cols + x];
                    // Invert y
                    colors[(set.Cols - 1 - y) * set.Cols + x] = new Color(pix, pix, pix, 1f);
                }
            }

            tex.SetPixels(0, 0, set.Rows, set.Cols, colors);
            tex.Apply(false);
        }

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
    }

    public class Batch {
        public int[] Indices;

        public Batch(int size) {
            Indices = new int[size];
        }
    }
}