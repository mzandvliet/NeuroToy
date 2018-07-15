using System.IO;
using System.Text;
using UnityEngine;
using System.Collections.Generic;

namespace NNClassic {
    public struct Dataset {
        public int[] Labels;
        public float[,] Images;
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
            Labels = new int[count];
            Images = new float[count, rows * cols];
            Rows = rows;
            Cols = cols;
            Indices = new List<int>(count);
        }
    }

    public static class Mnist {
        private const string Folder = "./Datasets/Mnist";
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

            Debug.Log("Loading " + imgPath + ", Imgs: " + NumImgs + " Rows: " + Rows + " Cols: " + Cols);

            var set = new Dataset(NumImgs, Rows, Cols);

            for (int i = 0; i < NumImgs; i++) {
                byte lbl = lblReader.ReadByte();
                set.Labels[i] = (int)lbl;
            }

            for (int i = 0; i < NumImgs; i++) {
                for (int j = 0; j < ImgDims; j++) {
                    byte pix = imgReader.ReadByte();
                    set.Images[i, j] = pix / 256f;
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
                    float pix = set.Images[imgIndex, y * set.Cols + x];
                    // Invert y
                    colors[(set.Cols - 1 - y) * set.Cols + x] = new Color(pix, pix, pix, 1f);
                }
            }

            tex.SetPixels(0, 0, set.Rows, set.Cols, colors);
            tex.Apply(false);
        }

       
    }

    public class Batch {
        public int[] Indices;

        public Batch(int size) {
            Indices = new int[size];
        }
    }
}